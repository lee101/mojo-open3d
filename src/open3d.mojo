"""Compute kernels for mojo-open3d.

All storage belongs to the caller. Buffers cross the C ABI as integer
addresses and are rebuilt with a concrete mutable origin inside each export.
"""

from std.math import floor, sqrt
from std.sys.info import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime KDTREE_BUILD_PARALLEL_THRESHOLD = 16384
comptime KDTREE_SEARCH_PARALLEL_THRESHOLD = 1024
comptime KDTREE_SEARCH_CHUNK_SIZE = 256


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def point_distance2(points: FPtr, point_idx: Int, query: FPtr) -> Float64:
    var base = point_idx * 3
    var dx = points[base] - query[0]
    var dy = points[base + 1] - query[1]
    var dz = points[base + 2] - query[2]
    return dx * dx + dy * dy + dz * dz


def fill_search_results(
    result_idx: IPtr,
    result_dist: FPtr,
    limit: Int,
    max_distance2: Float64,
):
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    var index_fill = SIMD[DType.int64, W](-1)
    var distance_fill = SIMD[DType.float64, W](max_distance2)
    while i + W <= limit:
        result_idx.store(i, index_fill)
        result_dist.store(i, distance_fill)
        i += W
    while i < limit:
        result_idx[i] = -1
        result_dist[i] = max_distance2
        i += 1


def less_on_axis(points: FPtr, a: Int, b: Int, axis: Int) -> Bool:
    var av = points[a * 3 + axis]
    var bv = points[b * 3 + axis]
    return av < bv or (av == bv and a < b)


def swap_indices(indices: IPtr, a: Int, b: Int):
    var tmp = indices[a]
    indices[a] = indices[b]
    indices[b] = tmp


def partition(
    points: FPtr, indices: IPtr, left: Int, right: Int, pivot: Int, axis: Int
) -> Int:
    var pivot_id = Int(indices[pivot])
    swap_indices(indices, pivot, right)
    var store = left
    for i in range(left, right):
        if less_on_axis(points, Int(indices[i]), pivot_id, axis):
            swap_indices(indices, store, i)
            store += 1
    swap_indices(indices, store, right)
    return store


def quickselect(
    points: FPtr, indices: IPtr, left_: Int, right_: Int, kth: Int, axis: Int
):
    var left = left_
    var right = right_
    while left < right:
        var pivot = left + (right - left) // 2
        pivot = partition(points, indices, left, right, pivot, axis)
        if pivot == kth:
            return
        if kth < pivot:
            right = pivot - 1
        else:
            left = pivot + 1


def build_node(
    points: FPtr,
    indices: IPtr,
    nodes: IPtr,
    axes: IPtr,
    left: Int,
    right: Int,
    depth: Int,
    node: Int,
):
    if left >= right:
        return
    var axis = depth % 3
    var middle = left + (right - left) // 2
    quickselect(points, indices, left, right - 1, middle, axis)
    nodes[node] = indices[middle]
    axes[node] = Int64(axis)
    build_node(points, indices, nodes, axes, left, middle, depth + 1, node * 2 + 1)
    build_node(
        points, indices, nodes, axes, middle + 1, right, depth + 1, node * 2 + 2
    )


def insert_neighbor(
    point_idx: Int,
    distance: Float64,
    result_idx: IPtr,
    result_dist: FPtr,
    count: Int,
    limit: Int,
) -> Int:
    var new_count = count
    if limit <= 0:
        return new_count
    if new_count == limit and distance >= result_dist[limit - 1]:
        return new_count
    var pos = new_count if new_count < limit else limit - 1
    if new_count < limit:
        new_count += 1
    while pos > 0 and (
        distance < result_dist[pos - 1]
        or (distance == result_dist[pos - 1] and point_idx < Int(result_idx[pos - 1]))
    ):
        if pos < limit:
            result_dist[pos] = result_dist[pos - 1]
            result_idx[pos] = result_idx[pos - 1]
        pos -= 1
    result_dist[pos] = distance
    result_idx[pos] = Int64(point_idx)
    return new_count


def search_node(
    points: FPtr,
    nodes: IPtr,
    axes: IPtr,
    node_capacity: Int,
    node: Int,
    query: FPtr,
    max_distance2: Float64,
    limit: Int,
    result_idx: IPtr,
    result_dist: FPtr,
    count_: Int,
) -> Int:
    if node >= node_capacity or nodes[node] < 0:
        return count_
    var count = count_
    var point_idx = Int(nodes[node])
    var axis = Int(axes[node])
    var delta = query[axis] - points[point_idx * 3 + axis]
    var near_node = node * 2 + 1 if delta <= 0.0 else node * 2 + 2
    var far_node = node * 2 + 2 if delta <= 0.0 else node * 2 + 1
    count = search_node(
        points,
        nodes,
        axes,
        node_capacity,
        near_node,
        query,
        max_distance2,
        limit,
        result_idx,
        result_dist,
        count,
    )
    var distance = point_distance2(points, point_idx, query)
    if distance <= max_distance2:
        count = insert_neighbor(
            point_idx, distance, result_idx, result_dist, count, limit
        )
    var boundary = max_distance2
    if count == limit and result_dist[limit - 1] < boundary:
        boundary = result_dist[limit - 1]
    if delta * delta <= boundary:
        count = search_node(
            points,
            nodes,
            axes,
            node_capacity,
            far_node,
            query,
            max_distance2,
            limit,
            result_idx,
            result_dist,
            count,
        )
    return count


def search_query(
    points: FPtr,
    nodes: IPtr,
    axes: IPtr,
    node_capacity: Int,
    queries: FPtr,
    query_index: Int,
    limit: Int,
    max_distance2: Float64,
    result_indices: IPtr,
    result_distances: FPtr,
    result_counts: IPtr,
):
    var offset = query_index * limit
    fill_search_results(
        result_indices + offset,
        result_distances + offset,
        limit,
        max_distance2,
    )
    result_counts[query_index] = Int64(
        search_node(
            points,
            nodes,
            axes,
            node_capacity,
            0,
            queries + query_index * 3,
            max_distance2,
            limit,
            result_indices + offset,
            result_distances + offset,
            0,
        )
    )


def hash_coord(x: Int, y: Int, z: Int, capacity: Int) -> Int:
    var value = (x * 73856093) ^ (y * 19349663) ^ (z * 83492791)
    if value < 0:
        value = -value
    return value % capacity


def voxel_aggregate(
    points: FPtr,
    colors: FPtr,
    normals: FPtr,
    point_count: Int,
    voxel_size: Float64,
    origin_x: Float64,
    origin_y: Float64,
    origin_z: Float64,
    bounded: Bool,
    max_x: Int,
    max_y: Int,
    max_z: Int,
    has_colors: Bool,
    has_normals: Bool,
    slot_x: IPtr,
    slot_y: IPtr,
    slot_z: IPtr,
    slot_ids: IPtr,
    capacity: Int,
    result_points: FPtr,
    result_colors: FPtr,
    result_normals: FPtr,
    result_counts: IPtr,
    trace_ids: IPtr,
) -> Int:
    var result_count = 0
    for i in range(point_count):
        trace_ids[i] = -1
        var base = i * 3
        var vx = Int(floor((points[base] - origin_x) / voxel_size))
        var vy = Int(floor((points[base + 1] - origin_y) / voxel_size))
        var vz = Int(floor((points[base + 2] - origin_z) / voxel_size))
        if bounded and (
            vx < 0 or vy < 0 or vz < 0 or vx >= max_x or vy >= max_y or vz >= max_z
        ):
            continue
        var slot = hash_coord(vx, vy, vz, capacity)
        while slot_ids[slot] >= 0 and (
            Int(slot_x[slot]) != vx
            or Int(slot_y[slot]) != vy
            or Int(slot_z[slot]) != vz
        ):
            slot = (slot + 1) % capacity
        var result_id: Int
        if slot_ids[slot] < 0:
            result_id = result_count
            result_count += 1
            slot_x[slot] = Int64(vx)
            slot_y[slot] = Int64(vy)
            slot_z[slot] = Int64(vz)
            slot_ids[slot] = Int64(result_id)
            result_counts[result_id] = 0
            for j in range(3):
                result_points[result_id * 3 + j] = 0.0
                result_colors[result_id * 3 + j] = 0.0
                result_normals[result_id * 3 + j] = 0.0
        else:
            result_id = Int(slot_ids[slot])
        trace_ids[i] = Int64(result_id)
        result_counts[result_id] += 1
        for j in range(3):
            result_points[result_id * 3 + j] += points[base + j]
            if has_colors:
                result_colors[result_id * 3 + j] += colors[base + j]
            if has_normals:
                result_normals[result_id * 3 + j] += normals[base + j]
    for i in range(result_count):
        var inv_count = 1.0 / Float64(result_counts[i])
        for j in range(3):
            result_points[i * 3 + j] *= inv_count
            if has_colors:
                result_colors[i * 3 + j] *= inv_count
            if has_normals:
                result_normals[i * 3 + j] *= inv_count
    return result_count


def identity4(matrix: FPtr):
    for i in range(16):
        matrix[i] = 0.0
    matrix[0] = 1.0
    matrix[5] = 1.0
    matrix[10] = 1.0
    matrix[15] = 1.0


def rigid_estimate(
    source: FPtr,
    target: FPtr,
    target_indices: IPtr,
    source_count: Int,
    with_scaling: Bool,
    matrix: FPtr,
) -> Int:
    identity4(matrix)
    var count = 0
    var sx = 0.0
    var sy = 0.0
    var sz = 0.0
    var tx = 0.0
    var ty = 0.0
    var tz = 0.0
    for i in range(source_count):
        var j = Int(target_indices[i])
        if j < 0:
            continue
        count += 1
        sx += source[i * 3]
        sy += source[i * 3 + 1]
        sz += source[i * 3 + 2]
        tx += target[j * 3]
        ty += target[j * 3 + 1]
        tz += target[j * 3 + 2]
    if count == 0:
        return 0
    var inv_count = 1.0 / Float64(count)
    sx *= inv_count
    sy *= inv_count
    sz *= inv_count
    tx *= inv_count
    ty *= inv_count
    tz *= inv_count
    if count < 3:
        matrix[3] = tx - sx
        matrix[7] = ty - sy
        matrix[11] = tz - sz
        return count
    var sxx = 0.0
    var sxy = 0.0
    var sxz = 0.0
    var syx = 0.0
    var syy = 0.0
    var syz = 0.0
    var szx = 0.0
    var szy = 0.0
    var szz = 0.0
    var source_energy = 0.0
    for i in range(source_count):
        var j = Int(target_indices[i])
        if j < 0:
            continue
        var px = source[i * 3] - sx
        var py = source[i * 3 + 1] - sy
        var pz = source[i * 3 + 2] - sz
        var qx = target[j * 3] - tx
        var qy = target[j * 3 + 1] - ty
        var qz = target[j * 3 + 2] - tz
        sxx += px * qx
        sxy += px * qy
        sxz += px * qz
        syx += py * qx
        syy += py * qy
        syz += py * qz
        szx += pz * qx
        szy += pz * qy
        szz += pz * qz
        source_energy += px * px + py * py + pz * pz
    var n00 = sxx + syy + szz
    var n01 = syz - szy
    var n02 = szx - sxz
    var n03 = sxy - syx
    var n11 = sxx - syy - szz
    var n12 = sxy + syx
    var n13 = szx + sxz
    var n22 = -sxx + syy - szz
    var n23 = syz + szy
    var n33 = -sxx - syy + szz
    var shift = (
        abs(n00) + abs(n01) + abs(n02) + abs(n03) + abs(n11) + abs(n12)
        + abs(n13) + abs(n22) + abs(n23) + abs(n33) + 1.0
    )
    var qw = 1.0
    var qx = 0.0
    var qy = 0.0
    var qz = 0.0
    for _ in range(80):
        var rw = (n00 + shift) * qw + n01 * qx + n02 * qy + n03 * qz
        var rx = n01 * qw + (n11 + shift) * qx + n12 * qy + n13 * qz
        var ry = n02 * qw + n12 * qx + (n22 + shift) * qy + n23 * qz
        var rz = n03 * qw + n13 * qx + n23 * qy + (n33 + shift) * qz
        var norm = sqrt(rw * rw + rx * rx + ry * ry + rz * rz)
        if norm == 0.0:
            break
        qw = rw / norm
        qx = rx / norm
        qy = ry / norm
        qz = rz / norm
    var r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    var r01 = 2.0 * (qx * qy - qz * qw)
    var r02 = 2.0 * (qx * qz + qy * qw)
    var r10 = 2.0 * (qx * qy + qz * qw)
    var r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    var r12 = 2.0 * (qy * qz - qx * qw)
    var r20 = 2.0 * (qx * qz - qy * qw)
    var r21 = 2.0 * (qy * qz + qx * qw)
    var r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
    var scale = 1.0
    if with_scaling and source_energy > 0.0:
        var numerator = 0.0
        for i in range(source_count):
            var j = Int(target_indices[i])
            if j < 0:
                continue
            var px = source[i * 3] - sx
            var py = source[i * 3 + 1] - sy
            var pz = source[i * 3 + 2] - sz
            var qxx = target[j * 3] - tx
            var qyy = target[j * 3 + 1] - ty
            var qzz = target[j * 3 + 2] - tz
            numerator += (
                qxx * (r00 * px + r01 * py + r02 * pz)
                + qyy * (r10 * px + r11 * py + r12 * pz)
                + qzz * (r20 * px + r21 * py + r22 * pz)
            )
        scale = numerator / source_energy
    matrix[0] = scale * r00
    matrix[1] = scale * r01
    matrix[2] = scale * r02
    matrix[4] = scale * r10
    matrix[5] = scale * r11
    matrix[6] = scale * r12
    matrix[8] = scale * r20
    matrix[9] = scale * r21
    matrix[10] = scale * r22
    matrix[3] = tx - scale * (r00 * sx + r01 * sy + r02 * sz)
    matrix[7] = ty - scale * (r10 * sx + r11 * sy + r12 * sz)
    matrix[11] = tz - scale * (r20 * sx + r21 * sy + r22 * sz)
    return count


def kdtree_build(
    points_addr: Int,
    point_count: Int,
    indices_addr: Int,
    nodes_addr: Int,
    axes_addr: Int,
    node_capacity: Int,
    parallel_enabled: Bool,
):
    var points = fp(points_addr)
    var indices = ip(indices_addr)
    var nodes = ip(nodes_addr)
    var axes = ip(axes_addr)
    for i in range(point_count):
        indices[i] = Int64(i)
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    var empty_nodes = SIMD[DType.int64, W](-1)
    while i + W <= node_capacity:
        nodes.store(i, empty_nodes)
        axes.store(i, empty_nodes)
        i += W
    while i < node_capacity:
        nodes[i] = -1
        axes[i] = -1
        i += 1
    if point_count > 0:
        if point_count < KDTREE_BUILD_PARALLEL_THRESHOLD or not parallel_enabled:
            build_node(points, indices, nodes, axes, 0, point_count, 0, 0)
        else:
            var m0 = point_count // 2
            quickselect(points, indices, 0, point_count - 1, m0, 0)
            nodes[0] = indices[m0]
            axes[0] = 0

            var m1 = m0 // 2
            quickselect(points, indices, 0, m0 - 1, m1, 1)
            nodes[1] = indices[m1]
            axes[1] = 1
            var m2 = m0 + 1 + (point_count - m0 - 1) // 2
            quickselect(points, indices, m0 + 1, point_count - 1, m2, 1)
            nodes[2] = indices[m2]
            axes[2] = 1

            var m3 = m1 // 2
            quickselect(points, indices, 0, m1 - 1, m3, 2)
            nodes[3] = indices[m3]
            axes[3] = 2
            var m4 = m1 + 1 + (m0 - m1 - 1) // 2
            quickselect(points, indices, m1 + 1, m0 - 1, m4, 2)
            nodes[4] = indices[m4]
            axes[4] = 2
            var m5 = m0 + 1 + (m2 - m0 - 1) // 2
            quickselect(points, indices, m0 + 1, m2 - 1, m5, 2)
            nodes[5] = indices[m5]
            axes[5] = 2
            var m6 = m2 + 1 + (point_count - m2 - 1) // 2
            quickselect(points, indices, m2 + 1, point_count - 1, m6, 2)
            nodes[6] = indices[m6]
            axes[6] = 2

            @__parameter
            def build_subtree(task: Int) capturing -> None:
                var task_points = fp(points_addr)
                var task_indices = ip(indices_addr)
                var task_nodes = ip(nodes_addr)
                var task_axes = ip(axes_addr)
                if task == 0:
                    build_node(
                        task_points, task_indices, task_nodes, task_axes, 0, m3, 3, 7
                    )
                elif task == 1:
                    build_node(
                        task_points,
                        task_indices,
                        task_nodes,
                        task_axes,
                        m3 + 1,
                        m1,
                        3,
                        8,
                    )
                elif task == 2:
                    build_node(
                        task_points,
                        task_indices,
                        task_nodes,
                        task_axes,
                        m1 + 1,
                        m4,
                        3,
                        9,
                    )
                elif task == 3:
                    build_node(
                        task_points,
                        task_indices,
                        task_nodes,
                        task_axes,
                        m4 + 1,
                        m0,
                        3,
                        10,
                    )
                elif task == 4:
                    build_node(
                        task_points,
                        task_indices,
                        task_nodes,
                        task_axes,
                        m0 + 1,
                        m5,
                        3,
                        11,
                    )
                elif task == 5:
                    build_node(
                        task_points,
                        task_indices,
                        task_nodes,
                        task_axes,
                        m5 + 1,
                        m2,
                        3,
                        12,
                    )
                elif task == 6:
                    build_node(
                        task_points,
                        task_indices,
                        task_nodes,
                        task_axes,
                        m2 + 1,
                        m6,
                        3,
                        13,
                    )
                else:
                    build_node(
                        task_points,
                        task_indices,
                        task_nodes,
                        task_axes,
                        m6 + 1,
                        point_count,
                        3,
                        14,
                    )

            for task in range(8):
                build_subtree(task)


@export("m3d_kdtree_build")
def m3d_kdtree_build(
    points_addr: Int,
    point_count: Int,
    indices_addr: Int,
    nodes_addr: Int,
    axes_addr: Int,
    node_capacity: Int,
    parallel_enabled: Int,
) abi("C"):
    kdtree_build(
        points_addr,
        point_count,
        indices_addr,
        nodes_addr,
        axes_addr,
        node_capacity,
        parallel_enabled != 0,
    )


def kdtree_search(
    points_addr: Int,
    nodes_addr: Int,
    axes_addr: Int,
    node_capacity: Int,
    queries_addr: Int,
    query_count: Int,
    limit: Int,
    max_distance2: Float64,
    result_indices_addr: Int,
    result_distances_addr: Int,
    result_counts_addr: Int,
    parallel_enabled: Bool,
):
    var points = fp(points_addr)
    var nodes = ip(nodes_addr)
    var axes = ip(axes_addr)
    var queries = fp(queries_addr)
    var result_indices = ip(result_indices_addr)
    var result_distances = fp(result_distances_addr)
    var result_counts = ip(result_counts_addr)
    if query_count < KDTREE_SEARCH_PARALLEL_THRESHOLD or not parallel_enabled:
        for q in range(query_count):
            search_query(
                points,
                nodes,
                axes,
                node_capacity,
                queries,
                q,
                limit,
                max_distance2,
                result_indices,
                result_distances,
                result_counts,
            )
    else:
        var chunk_count = (
            query_count + KDTREE_SEARCH_CHUNK_SIZE - 1
        ) // KDTREE_SEARCH_CHUNK_SIZE

        @__parameter
        def search_chunk(chunk: Int) capturing -> None:
            var task_points = fp(points_addr)
            var task_nodes = ip(nodes_addr)
            var task_axes = ip(axes_addr)
            var task_queries = fp(queries_addr)
            var task_result_indices = ip(result_indices_addr)
            var task_result_distances = fp(result_distances_addr)
            var task_result_counts = ip(result_counts_addr)
            var start = chunk * KDTREE_SEARCH_CHUNK_SIZE
            var end = min(start + KDTREE_SEARCH_CHUNK_SIZE, query_count)
            for q in range(start, end):
                search_query(
                    task_points,
                    task_nodes,
                    task_axes,
                    node_capacity,
                    task_queries,
                    q,
                    limit,
                    max_distance2,
                    task_result_indices,
                    task_result_distances,
                    task_result_counts,
                )

        for chunk in range(chunk_count):
            search_chunk(chunk)


@export("m3d_kdtree_search")
def m3d_kdtree_search(
    points_addr: Int,
    nodes_addr: Int,
    axes_addr: Int,
    node_capacity: Int,
    queries_addr: Int,
    query_count: Int,
    limit: Int,
    max_distance2: Float64,
    result_indices_addr: Int,
    result_distances_addr: Int,
    result_counts_addr: Int,
    parallel_enabled: Int,
) abi("C"):
    kdtree_search(
        points_addr,
        nodes_addr,
        axes_addr,
        node_capacity,
        queries_addr,
        query_count,
        limit,
        max_distance2,
        result_indices_addr,
        result_distances_addr,
        result_counts_addr,
        parallel_enabled != 0,
    )


@export("m3d_voxel_aggregate")
def m3d_voxel_aggregate(
    points_addr: Int,
    colors_addr: Int,
    normals_addr: Int,
    point_count: Int,
    voxel_size: Float64,
    origin_x: Float64,
    origin_y: Float64,
    origin_z: Float64,
    bounded: Int,
    max_x: Int,
    max_y: Int,
    max_z: Int,
    has_colors: Int,
    has_normals: Int,
    slot_x_addr: Int,
    slot_y_addr: Int,
    slot_z_addr: Int,
    slot_ids_addr: Int,
    capacity: Int,
    result_points_addr: Int,
    result_colors_addr: Int,
    result_normals_addr: Int,
    result_counts_addr: Int,
    trace_ids_addr: Int,
) abi("C") -> Int:
    return voxel_aggregate(
        fp(points_addr),
        fp(colors_addr),
        fp(normals_addr),
        point_count,
        voxel_size,
        origin_x,
        origin_y,
        origin_z,
        bounded != 0,
        max_x,
        max_y,
        max_z,
        has_colors != 0,
        has_normals != 0,
        ip(slot_x_addr),
        ip(slot_y_addr),
        ip(slot_z_addr),
        ip(slot_ids_addr),
        capacity,
        fp(result_points_addr),
        fp(result_colors_addr),
        fp(result_normals_addr),
        ip(result_counts_addr),
        ip(trace_ids_addr),
    )


@export("m3d_transform_points")
def m3d_transform_points(
    points_addr: Int, matrix_addr: Int, result_addr: Int, point_count: Int
) abi("C"):
    var points = fp(points_addr)
    var matrix = fp(matrix_addr)
    var result = fp(result_addr)
    for i in range(point_count):
        var x = points[i * 3]
        var y = points[i * 3 + 1]
        var z = points[i * 3 + 2]
        var w = matrix[12] * x + matrix[13] * y + matrix[14] * z + matrix[15]
        if w == 0.0:
            w = 1.0
        result[i * 3] = (
            matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3]
        ) / w
        result[i * 3 + 1] = (
            matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7]
        ) / w
        result[i * 3 + 2] = (
            matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11]
        ) / w


@export("m3d_rigid_estimate")
def m3d_rigid_estimate(
    source_addr: Int,
    target_addr: Int,
    target_indices_addr: Int,
    source_count: Int,
    with_scaling: Int,
    matrix_addr: Int,
) abi("C") -> Int:
    return rigid_estimate(
        fp(source_addr),
        fp(target_addr),
        ip(target_indices_addr),
        source_count,
        with_scaling != 0,
        fp(matrix_addr),
    )
