import taichi as ti
import math
import numpy as np

ti.init(arch=ti.gpu)  # or ti.cpu

res = (800, 800)
samples_per_pixel = 50
max_depth = 64

vec3 = ti.types.vector(3, float)

@ti.dataclass
class Ray:
    origin: vec3
    direction: vec3

@ti.dataclass
class Sphere:
    center: vec3
    radius: float
    color: vec3
    is_light: int
    material: int  # 0: diffuse, 1: metal, 2: glass

@ti.dataclass
class Plane:
    point: vec3
    normal: vec3
    color: vec3
    is_light: int
    material: int
    has_texture:int
    texture_scale:float

n_spheres = 5
spheres = Sphere.field(shape=n_spheres)

n_planes = 10
planes = Plane.field(shape=n_planes)

image = ti.Vector.field(3, dtype=float, shape=res)

@ti.func
def reflect(v, n):
    return v - 2 * v.dot(n) * n

@ti.func
def refract(v, n, ni_over_nt, refracted):
    uv = v.normalized()
    dt = uv.dot(n)
    discriminant = 1.0 - ni_over_nt * ni_over_nt * (1 - dt * dt)

    if discriminant > 0:
        refracted[0] = ni_over_nt * (uv - n * dt) - n * ti.sqrt(discriminant)
        # success = True
        return True
    return False

@ti.func
def schlick(cosine, ref_idx):
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0 * r0
    return r0 + (1 - r0) * pow((1 - cosine), 5)

@ti.func
def random_in_unit_sphere():
    p = vec3(0.0, 0.0, 0.0)
    found = False
    for _ in range(100):
        sample = vec3(ti.random(), ti.random(), ti.random()) * 2.0 - vec3(1.0, 1.0, 1.0)
        if sample.norm_sqr() < 1.0 and not found:
            p = sample
            found = True
    return p

@ti.func
def intersect_sphere(ray: Ray, sphere: Sphere):
    oc = ray.origin - sphere.center
    a = ray.direction.dot(ray.direction)
    b = 2.0 * oc.dot(ray.direction)
    c = oc.dot(oc) - sphere.radius * sphere.radius
    discriminant = b * b - 4 * a * c
    hit = False
    t = 1e6
    if discriminant > 0:
        sqrtd = ti.sqrt(discriminant)
        temp = (-b - sqrtd) / (2.0 * a)
        if 1e-3 < temp < t:
            t = temp
            hit = True
        temp = (-b + sqrtd) / (2.0 * a)
        if 1e-3 < temp < t:
            t = temp
            hit = True
    return hit, t

@ti.func
def intersect_plane(ray: Ray, plane: Plane):
    hit = False
    t = 0.0
    tex_coord = vec3(0.0, 0.0, 0.0)  # 纹理坐标
    denom = plane.normal.dot(ray.direction)
    if ti.abs(denom) > 1e-6:
        t_candidate = (plane.point - ray.origin).dot(plane.normal) / denom
        if t_candidate > 1e-3:
            hit = True
            t = t_candidate
            hit_point = ray.origin + ray.direction * t
            
            # 计算纹理坐标
            if plane.has_texture:
                # 创建局部坐标系
                up = vec3(0, 1, 0) if ti.abs(plane.normal.dot(vec3(0, 1, 0))) < 0.99 else vec3(1, 0, 0)
                tangent = up.cross(plane.normal).normalized()
                bitangent = plane.normal.cross(tangent)
                
                # 投影到平面坐标系
                rel_pos = hit_point - plane.point
                u = rel_pos.dot(tangent) * plane.texture_scale
                v = rel_pos.dot(bitangent) * plane.texture_scale
                
                # 确保坐标在[0,1]范围内
                u = u - ti.floor(u)
                v = v - ti.floor(v)
                tex_coord = vec3(u, v, 0)
    
    return hit, t, tex_coord

@ti.func
def refract(v, n, ni_over_nt):
    uv = v.normalized()
    dt = uv.dot(n)
    discriminant = 1.0 - ni_over_nt * ni_over_nt * (1 - dt * dt)
    refracted = vec3(0.0)
    success = False
    
    if discriminant > 0:
        refracted = ni_over_nt * (uv - n * dt) - n * ti.sqrt(discriminant)
        success = True
    
    return success, refracted

@ti.func
def compute_ambient_occlusion(p, n, num_samples: int) -> float:
    occlusion = 0.0
    for sample in range(num_samples):
        dir = (n + random_in_unit_sphere()).normalized()
        origin = p + n * 1e-3
        ray = Ray(origin, dir)

        # 检测是否遮挡
        is_occluded = 0  # 用 int 标志
        for j in range(n_spheres):
            hit, _ = intersect_sphere(ray, spheres[ti.cast(j, ti.i32)])
            if hit:
                is_occluded = 1
        for j in range(n_planes):
            hit, _, _ = intersect_plane(ray, planes[ti.cast(j, ti.i32)])
            if hit:
                is_occluded = 1
        occlusion += is_occluded

    return 1.0 - occlusion / num_samples

@ti.func
def lerp(a, b, t):
    return a * (1 - t) + b * t


@ti.func
def trace(ray: Ray, depth: int) -> vec3:
    color = vec3(0.0, 0.0, 0.0)
    attenuation = vec3(1.0, 1.0, 1.0)

    for _ in range(depth):
        closest_t = 1e6
        hit_any = False
        hit_point = vec3(0.0, 0.0, 0.0)
        normal = vec3(0.0, 0.0, 0.0)
        mat = 0
        hit_color = vec3(0.0, 0.0, 0.0)
        is_light = 0
        tex_coord = vec3(0.0, 0.0, 0.0)

        # Sphere hit
        for i in range(n_spheres):
            hit, t = intersect_sphere(ray, spheres[i])
            if hit and t < closest_t:
                closest_t = t
                hit_any = True
                hit_point = ray.origin + ray.direction * t
                normal = (hit_point - spheres[i].center).normalized()
                hit_color = spheres[i].color
                mat = spheres[i].material
                is_light = spheres[i].is_light

        # Plane hit - 修改后的部分
        for i in range(n_planes):
            hit, t, coord = intersect_plane(ray, planes[i])  # 接收三个返回值
            if hit and t < closest_t:
                closest_t = t
                hit_any = True
                hit_point = ray.origin + ray.direction * t
                normal = planes[i].normal
                hit_color = planes[i].color
                mat = planes[i].material
                is_light = planes[i].is_light
                tex_coord = coord
                
                if planes[i].has_texture:
                    u = int(tex_coord.x * (texture_res[0]-1))
                    v = int(tex_coord.y * (texture_res[1]-1))
                    hit_color *= checker_texture[u, v]

        if hit_any:
            if is_light:
                color += attenuation * hit_color
                break

            if mat == 0:  # Diffuse
                target = hit_point + normal + random_in_unit_sphere()
                ray = Ray(hit_point + normal * 1e-4, (target - hit_point).normalized())
                attenuation *= hit_color
                
                
                # ao = compute_ambient_occlusion(hit_point, normal, num_samples=32)
                # ao = max(ao,0.2)
                # attenuation *= lerp(hit_color * ao, hit_color, 0.5)

            elif mat == 1:  # Metal
                reflected = reflect(ray.direction.normalized(), normal)
                ray = Ray(hit_point + normal * 1e-4, reflected.normalized())
                attenuation *= hit_color
            elif mat == 2:  # Glass (refraction)
                reflected = reflect(ray.direction.normalized(), normal)
                
                outward_normal = normal  # 默认值
                ior = 2.0
                ni_over_nt = 1.0 / ior  # 默认值（从玻璃到空气）
                
                entering = ray.direction.dot(normal) < 0
                if entering:
                    ni_over_nt = 1.0 / 1.5  # 从空气到玻璃
                    outward_normal = normal
                else:
                    outward_normal = -normal
                
                can_refract, refracted = refract(ray.direction.normalized(), 
                                              outward_normal, 
                                              ni_over_nt)
                
                cosine = min(abs(ray.direction.normalized().dot(normal)), 1.0)
                reflect_prob = schlick(cosine, ior if entering else 1.0/ior)
                
                if not can_refract:
                    reflect_prob = 1.0
                
                if ti.random() < reflect_prob:
                    ray = Ray(hit_point + outward_normal * 1e-4, reflected)
                else:
                    ray = Ray(hit_point - outward_normal * 1e-4, refracted)

        else:
            break

    return color


# 在ti.init之后添加纹理相关代码
texture_res = (512, 512)
checker_texture = ti.Vector.field(3, dtype=float, shape=texture_res)

@ti.kernel
def init_checker_texture():
    for i, j in checker_texture:
        # 创建棋盘格纹理
        tile_size = 32
        if (i // tile_size + j // tile_size) % 2 == 0:
            checker_texture[i, j] = vec3(0.8, 0.8, 0.8)  # 浅色
        else:
            checker_texture[i, j] = vec3(0.3, 0.3, 0.3)  # 深色

@ti.kernel
def render():
    for i, j in image:
        col = vec3(0.0, 0.0, 0.0)
        for s in range(samples_per_pixel):
            u = (i + ti.random()) / res[0]
            v = (j + ti.random()) / res[1]

            # Camera setup
            lookfrom = vec3(0.0, 1.0, -9.0)
            lookat = vec3(0.0, 1.0, 0.0)
            up = vec3(0.0, 1.0, 0.0)
            fov = 60.0
            aspect = res[0] / res[1]

            theta = fov * ti.math.pi / 180.0
            half_height = ti.tan(theta / 2)
            half_width = aspect * half_height

            w = (lookfrom - lookat).normalized()
            u_cam = up.cross(w).normalized()
            v_cam = w.cross(u_cam)

            dir = -w + u_cam * (2 * u - 1) * half_width + v_cam * (2 * v - 1) * half_height
            ray = Ray(lookfrom, dir.normalized())

            col += trace(ray, max_depth)

        col /= samples_per_pixel
        col = vec3(ti.sqrt(col[0]), ti.sqrt(col[1]), ti.sqrt(col[2]))  # gamma correction
        image[i, j] = col

def main():
    init_checker_texture()
    # 场景物体
    spheres[0] = Sphere(vec3(-3, 1.0, 0), 0.8, vec3(0.8, 0.3, 0.3), 0, 0)  # 红色漫反射
    spheres[1] = Sphere(vec3(3, 1.0, 0), 1, vec3(0.95, 0.95, 0.95), 0, 1)  # 金属
    
    spheres[2] = Sphere(vec3(0, 1.0, -2), 1, vec3(0.9, 0.9, 0.9), 0, 2)  # 玻璃球
    # spheres[3] = Sphere(vec3(0,1.0, -4), 1, vec3(0.3, 0.5, 0.24), 0, 0)     
    # 光源/平面
    planes[0] = Plane(vec3(0, 0, 0), vec3(0, 1, 0), vec3(1, 1, 1), 0, 0, 1, 0.2)  # 地板


    planes[1] = Plane(vec3(0, 5, 0), vec3(0, -1, 0), vec3(0.5, 0.5, 0.5), 1, 0)      # 顶部光源
    


    # 左墙（垂直于x轴，朝向x正方向）
    planes[2] = Plane(vec3(-4, 0, 0), vec3(1, 0, 0), vec3(0.8, 0.6, 0.6), 0, 0)
    # 右墙（垂直于x轴，朝向x负方向）
    planes[3] = Plane(vec3(4, 0, 0), vec3(-1, 0, 0), vec3(0.6, 0.8, 0.6), 0, 0)
    # 后墙（垂直于z轴，朝向z正方向）
    planes[4] = Plane(vec3(0, 0, 4), vec3(0, 0, -1), vec3(0.6, 0.6, 0.8), 0, 0)
    

    render()

    # 窗口展示
    window = ti.ui.Window("Taichi Path Tracer", res)
    canvas = window.get_canvas()
    while window.running:
        canvas.set_image(image)
        window.show()

if __name__ == "__main__":
    main()