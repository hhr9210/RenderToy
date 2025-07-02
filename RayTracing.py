import numpy as np
import matplotlib.pyplot as plt
import time
import math
import cv2
from concurrent.futures import ProcessPoolExecutor
from scipy.ndimage import gaussian_filter

# 全局变量用于进程共享
global_scene = None
global_camera = None

def normalize(v):
    return v / np.linalg.norm(v)

def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

def refracted(vector, normal, ior):
    cosi = -np.dot(normal, vector)
    etai = 1
    etat = ior
    if cosi < 0:
        cosi = -cosi
        normal = -normal
        etai, etat = etat, etai
    eta = etai / etat
    k = 1 - eta**2 * (1 - cosi**2)
    if k < 0:
        return None
    else:
        return eta * vector + (eta * cosi - np.sqrt(k)) * normal

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = normalize(direction)

class Material:
    def __init__(self, color, diffuse=1.0, specular=0.5, reflection=0.0,
                 transparency=0.0, ior=1.0, roughness=0.5, metallic=0.0):
        self.color = np.array(color)
        self.diffuse = diffuse
        self.specular = specular
        self.reflection = reflection
        self.transparency = transparency
        self.ior = ior
        self.roughness = roughness
        self.metallic = metallic

class CheckerboardMaterial:
    def __init__(self, color1, color2, scale=1.0, roughness=1.0, diffuse=1.0, 
                 specular=0.5, reflection=0.0, transparency=0.0, ior=1.0, metallic=0.0):
        self.color1 = np.array(color1)
        self.color2 = np.array(color2)
        self.scale = scale
        self.roughness = roughness
        self.diffuse = diffuse
        self.specular = specular
        self.reflection = reflection  # 添加反射属性
        self.transparency = transparency
        self.ior = ior
        self.metallic = metallic

    def get_color(self, point):
        x, _, z = point
        x = math.floor(x * self.scale)
        z = math.floor(z * self.scale)
        if (x + z) % 2 == 0:
            return self.color1
        else:
            return self.color2

class Sphere:
    def __init__(self, center, radius, material):
        self.center = np.array(center)
        self.radius = radius
        self.material = material

    def intersect(self, ray):
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius**2
        discriminant = b**2 - 4*a*c
        if discriminant > 0:
            t1 = (-b - np.sqrt(discriminant)) / (2.0*a)
            t2 = (-b + np.sqrt(discriminant)) / (2.0*a)
            if t1 > 0.001:
                return t1
            elif t2 > 0.001:
                return t2
        return None

    def normal(self, point):
        return normalize(point - self.center)

class Plane:
    def __init__(self, point, normal, material):
        self.point = np.array(point)
        self._normal = normalize(np.array(normal))
        self.material = material

    def intersect(self, ray):
        denom = np.dot(self._normal, ray.direction)
        if abs(denom) > 0.0001:
            t = np.dot(self.point - ray.origin, self._normal) / denom
            if t > 0.001:
                return t
        return None

    def normal(self, point):
        return self._normal

    def get_color(self, point):
        if hasattr(self.material, 'get_color'):
            return self.material.get_color(point)
        return self.material.color

class Triangle:
    def __init__(self, v0, v1, v2, material):
        self.v0 = np.array(v0)
        self.v1 = np.array(v1)
        self.v2 = np.array(v2)
        self.material = material
        self.n = normalize(np.cross(self.v1 - self.v0, self.v2 - self.v0))

    def intersect(self, ray):
        e1 = self.v1 - self.v0
        e2 = self.v2 - self.v0
        h = np.cross(ray.direction, e2)
        a = np.dot(e1, h)
        if abs(a) < 1e-6:
            return None
        f = 1.0 / a
        s = ray.origin - self.v0
        u = f * np.dot(s, h)
        if u < 0.0 or u > 1.0:
            return None
        q = np.cross(s, e1)
        v = f * np.dot(ray.direction, q)
        if v < 0.0 or u + v > 1.0:
            return None
        t = f * np.dot(e2, q)
        if t > 0.001:
            return t
        return None

    def normal(self, point):
        return self.n

class Light:
    def __init__(self, position, color, intensity, radius=0.2):
        self.position = np.array(position)
        self.color = np.array(color)
        self.intensity = intensity
        self.radius = radius

# GGX PBR functions
def distribution_ggx(N, H, roughness):
    a = roughness**2
    a2 = a * a
    NdotH = max(np.dot(N, H), 0.0)
    NdotH2 = NdotH * NdotH
    denom = NdotH2 * (a2 - 1) + 1
    denom = np.pi * denom * denom
    return a2 / denom

def geometry_schlick_ggx(NdotV, roughness):
    r = (roughness + 1)
    k = (r*r) / 8
    denom = NdotV * (1 - k) + k
    return NdotV / denom

def geometry_smith(N, V, L, roughness):
    NdotV = max(np.dot(N, V), 0.0)
    NdotL = max(np.dot(N, L), 0.0)
    ggx1 = geometry_schlick_ggx(NdotV, roughness)
    ggx2 = geometry_schlick_ggx(NdotL, roughness)
    return ggx1 * ggx2

def fresnel_schlick(cos_theta, F0):
    return F0 + (1 - F0) * (1 - cos_theta)**5

class Scene:
    def __init__(self):
        self.objects = []
        self.lights = []
        self.skybox = Skybox()

    def add_object(self, obj):
        self.objects.append(obj)

    def add_light(self, light):
        self.lights.append(light)

    def intersect(self, ray):
        closest_obj = None
        closest_t = float('inf')
        for obj in self.objects:
            t = obj.intersect(ray)
            if t is not None and t < closest_t:
                closest_t = t
                closest_obj = obj
        if closest_obj is None:
            return None, None, None
        hit_point = ray.origin + closest_t * ray.direction
        normal = closest_obj.normal(hit_point)
        return closest_obj, hit_point, normal

    def ambient_occlusion(self, point, normal, samples=8, max_dist=1.0):
        occlusion = 0
        for _ in range(samples):
            while True:
                dir = np.random.uniform(-1, 1, 3)
                if np.linalg.norm(dir) <= 1 and np.dot(dir, normal) > 0:
                    break
            dir = normalize(dir)
            sample_ray = Ray(point + 0.001 * normal, dir)
            for obj in self.objects:
                t = obj.intersect(sample_ray)
                if t is not None and t < max_dist:
                    occlusion += 1
                    break
        return 1 - occlusion / samples

    def trace(self, ray, depth=0, max_depth=4, shadow_samples=32, ao_samples=32):
        if depth > max_depth:
            return self.skybox.get_color(ray.direction)

        closest_obj, hit_point, N = self.intersect(ray)
        if closest_obj is None:
            return self.skybox.get_color(ray.direction)

        # 获取材质颜色（支持纹理）
        if hasattr(closest_obj, 'get_color'):
            material_color = closest_obj.get_color(hit_point)
        else:
            material_color = closest_obj.material.color

        V = normalize(-ray.direction)
        color = np.zeros(3)

        ao = self.ambient_occlusion(hit_point, N, samples=ao_samples)
        ambient = 0.15 * ao * material_color

        F0 = np.array([0.04, 0.04, 0.04])
        if closest_obj.material.metallic > 0:
            F0 = material_color

        color += ambient

        for light in self.lights:
            light_contrib = np.zeros(3)
            visible_count = 0
            for _ in range(shadow_samples):
                while True:
                    rand_dir = np.random.uniform(-1, 1, 3)
                    if np.linalg.norm(rand_dir) <= 1:
                        break
                light_pos = light.position + rand_dir * light.radius
                L = normalize(light_pos - hit_point)
                dist_to_light = np.linalg.norm(light_pos - hit_point)
                shadow_ray = Ray(hit_point + 0.001 * N, L)
                if not any(obj.intersect(shadow_ray) is not None and obj.intersect(shadow_ray) < dist_to_light for obj in self.objects):
                    visible_count += 1
                    H = normalize(V + L)
                    NdotL = max(np.dot(N, L), 0.0)
                    D = distribution_ggx(N, H, closest_obj.material.roughness)
                    G = geometry_smith(N, V, L, closest_obj.material.roughness)
                    F = fresnel_schlick(max(np.dot(H, V), 0.0), F0)
                    kD = (1 - F) * (1 - closest_obj.material.metallic)
                    diffuse = kD * material_color / np.pi
                    specular = (D * G * F) / (4 * max(np.dot(N, V), 0.001) * NdotL + 1e-6)
                    light_contrib += (diffuse + specular) * light.color * light.intensity * NdotL

            if shadow_samples > 0:
                light_contrib /= shadow_samples
            color += light_contrib

        reflect_color = np.zeros(3)
        refract_color = np.zeros(3)
        cos_theta = max(np.dot(N, V), 0.0)
        F = fresnel_schlick(cos_theta, F0)

        if closest_obj.material.reflection > 0:
            reflect_dir = reflected(ray.direction, N)
            reflect_ray = Ray(hit_point + 0.001 * N, reflect_dir)
            reflect_color = self.trace(reflect_ray, depth + 1, max_depth)
            color += reflect_color * closest_obj.material.reflection

        if closest_obj.material.reflection > 0 or closest_obj.material.transparency > 0:
            reflect_dir = reflected(ray.direction, N)
            reflect_ray = Ray(hit_point + 0.001 * N, reflect_dir)
            reflect_color = self.trace(reflect_ray, depth + 1, max_depth, shadow_samples, ao_samples)

        if closest_obj.material.transparency > 0:
            refr_dir = refracted(ray.direction, N, closest_obj.material.ior)
            if refr_dir is not None:
                refr_ray = Ray(hit_point - 0.001 * N, refr_dir)
                refract_color = self.trace(refr_ray, depth + 1, max_depth, shadow_samples, ao_samples)

        color = color * (1 - closest_obj.material.transparency)
        color += reflect_color * F * closest_obj.material.reflection
        color += refract_color * (1 - F) * closest_obj.material.transparency

        return np.clip(color, 0, 1)

class Camera:
    def __init__(self, position, look_at, up, fov):
        self.position = np.array(position)
        self.look_at = np.array(look_at)
        self.up = normalize(np.array(up))
        self.fov = fov

        self.forward = normalize(self.look_at - self.position)
        self.right = normalize(np.cross(self.forward, self.up))
        self.up = np.cross(self.right, self.forward)

    def get_ray(self, x, y, width, height):
        aspect = width / height
        px = (2 * ((x + 0.5) / width) - 1) * np.tan(self.fov / 2) * aspect
        py = (1 - 2 * ((y + 0.5) / height)) * np.tan(self.fov / 2)
        direction = normalize(self.forward + px * self.right + py * self.up)
        return Ray(self.position, direction)

class Skybox:
    def __init__(self, color_top=[0.7,0.8,1.0], color_bottom=[0.2,0.3,0.5]):
        self.color_top = np.array(color_top)
        self.color_bottom = np.array(color_bottom)

    def get_color(self, direction):
        y = normalize(direction)[1]
        t = (y + 1) * 0.5
        return (1-t)*self.color_bottom + t*self.color_top

def init_globals(scene, camera):
    global global_scene, global_camera
    global_scene = scene
    global_camera = camera

def render_row(y, width, height):
    row = np.zeros((width, 3))
    for x in range(width):
        ray = global_camera.get_ray(x, y, width, height)
        row[x] = global_scene.trace(ray)
    return y, row

def render(scene, camera, width, height, filename):
    image = np.zeros((height, width, 3))
    start_time = time.time()

    with ProcessPoolExecutor(initializer=init_globals, initargs=(scene, camera)) as executor:
        futures = [executor.submit(render_row, y, width, height) for y in range(height)]
        for future in futures:
            y, row = future.result()
            image[y] = row
            if (y + 1) % 10 == 0:
                print(f"渲染进度: {100*(y+1)/height:.1f}%", end='\r')

    image = cv2.bilateralFilter(image.astype(np.float32), d=3, sigmaColor=0.5, sigmaSpace=5)
    plt.imsave(filename, image)
    print(f"\n渲染完成: {filename}，耗时 {time.time()-start_time:.1f} 秒")

def create_scene():
    scene = Scene()
    # 棋盘格地板材质
    floor = CheckerboardMaterial(
        color1=[0.4, 0.4, 0.4],  # 浅色格子
        color2=[0.1, 0.1, 0.1],  # 深色格子
        scale=1.5,               # 控制格子大小
        roughness=1.0,
        reflection=0.3
    )
    
    wall = Material([0.2, 0.3, 1], reflection=0.1)
    mirror = Material([0.5, 0.5, 0.5], reflection=0.3, roughness=0.1)
    left_wall = Material([0.2, 0.8, 0.1], roughness=0.3)
    right_wall = Material([0.9, 0.2, 0.1], roughness=0.3)

    scene.add_object(Plane([0, -2, 0], [0, 1, 0], floor))
    scene.add_object(Plane([0, 0, -10], [0, 0, 1], wall))
    scene.add_object(Plane([-4, 0, 0], [1, 0, 0], left_wall))
    scene.add_object(Plane([4, 0, 0], [-1, 0, 0], right_wall))

    scene.add_object(Sphere([1, -0.5, -3], 1.5, mirror))
    scene.add_object(Sphere([-2, -1, -4], 1, Material([1, 0, 0])))

    scene.add_light(Light([5, 10, 5], [1, 1, 1], 3, radius=3.0))
    return scene

if __name__ == "__main__":
    width, height = 100, 112
    cam = Camera([0, 0, 10], [0, 0, 0], [0, 1, 0], np.radians(45))
    scene = create_scene()
    render(scene, cam, width, height, "./render_with_checkerboard.png")