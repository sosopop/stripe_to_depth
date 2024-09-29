import pygame
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import glm
import cv2

vertex_shader = """
#version 330 core
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
in vec3 position;
in vec3 normal;
out vec3 v_normal;
out vec3 frag_pos;
void main()
{
    v_normal = normalize(mat3(transpose(inverse(model))) * normal);
    vec4 world_position = model * vec4(position, 1.0);
    frag_pos = world_position.xyz;
    gl_Position = projection * view * model * vec4(position, 1.0);
}
"""

fragment_shader = """
#version 330 core
in vec3 v_normal;
in vec3 frag_pos;
uniform vec4 ambient_color;
uniform vec3 light_position;
uniform vec3 view_pos;
uniform sampler2D shadow_map;
uniform mat4 light_space_matrix;
out vec4 frag_color;

float shadow_factor(vec4 frag_pos_light_space)
{
    vec3 proj_coords = frag_pos_light_space.xyz / frag_pos_light_space.w;
    proj_coords = proj_coords * 0.5 + 0.5;

    if(proj_coords.z > 1.0 || proj_coords.x < 0.0 || proj_coords.x > 1.0 || proj_coords.y < 0.0 || proj_coords.y > 1.0)
        return 1.0;

    float shadow = 0.0;
    float bias = 0.005;
    int pcf_samples = 2;
    float pcf_radius = 1.0 / 576.0;

    for(int x = -pcf_samples; x <= pcf_samples; ++x)
    {
        for(int y = -pcf_samples; y <= pcf_samples; ++y)
        {
            float closest_depth = texture(shadow_map, proj_coords.xy + vec2(x, y) * pcf_radius).r;
            if(proj_coords.z > closest_depth + bias)
                shadow += 1.0;
        }
    }

    shadow /= float((2 * pcf_samples + 1) * (2 * pcf_samples + 1));

    return shadow;
}

void main()
{
    vec4 frag_pos_light_space = light_space_matrix * vec4(frag_pos, 1.0);
    float shadow = shadow_factor(frag_pos_light_space);
    
    vec3 norm = v_normal;
    vec3 light_dir = normalize(light_position - frag_pos);

    // 漫反射
    float diff = max(dot(norm, light_dir), 0.0);
    vec4 diffuse_color = vec4(diff, diff, diff, 1.0);

    // 镜面反射
    vec3 view_dir = normalize(view_pos - frag_pos);
    vec3 reflect_dir = reflect(-light_dir, norm);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32);
    vec4 specular = spec * vec4(1.0, 1.0, 1.0, 1.0);
    
    // 条纹效果
    vec3 stripe_dir = normalize(frag_pos - light_position);
    float stripe_pattern = (sin(dot(stripe_dir, vec3(0.0, 1.0, 0.0)) * 300) + 1.0) / 2.0;
    stripe_pattern = pow(stripe_pattern, 3.0);
    vec4 stripe_color = vec4(stripe_pattern, stripe_pattern, stripe_pattern, 1.0);

    // 光照颜色
    vec4 lighting_color = (diffuse_color + specular);
    vec4 final_color = mix(lighting_color, stripe_color, 0.5);

    frag_color = ambient_color + final_color * (1.0 - shadow);
}
"""

def get_world_depth_buffer(view_matrix, projection_matrix, screen_width, screen_height, near, far):
    # 1. 读取深度缓冲区
    depth_buffer = np.zeros((screen_height, screen_width), dtype=np.float32)
    glReadPixels(0, 0, screen_width, screen_height, GL_DEPTH_COMPONENT, GL_FLOAT, depth_buffer)

    # 2. 准备投影矩阵的逆矩阵和视图矩阵的逆矩阵
    inv_proj_matrix = glm.inverse(projection_matrix)
    inv_view_matrix = glm.inverse(view_matrix)
    
    # 3. 创建一个用于保存世界坐标系Z深度的矩阵
    world_z_depth = np.zeros((screen_height, screen_width), dtype=np.float32)

    # 4. 遍历每个像素并计算世界坐标系中的深度
    for y in range(screen_height):
        for x in range(screen_width):
            # 获取该像素的深度值
            depth = depth_buffer[y, x]
            
            # 将深度值从 [0, 1] 范围映射到 NDC (-1, 1)
            z_ndc = depth * 2.0 - 1.0
            
            # 将屏幕坐标转换为 NDC 坐标
            x_ndc = (2.0 * x / screen_width) - 1.0
            y_ndc = 1.0 - (2.0 * y / screen_height)

            # 计算相机坐标系中的坐标（透视除法后的逆操作）
            clip_space_pos = glm.vec4(x_ndc, y_ndc, z_ndc, 1.0)
            eye_space_pos = inv_proj_matrix * clip_space_pos
            
            # 进行透视除法得到相机坐标系中的坐标
            if eye_space_pos.w != 0.0:
                eye_space_pos /= eye_space_pos.w
            
            # 将相机坐标转换到世界坐标系
            world_space_pos = inv_view_matrix * eye_space_pos

            # 保存世界坐标系中的Z深度
            world_z_depth[y, x] = world_space_pos.z

    # 5. 返回世界坐标系的Z深度矩阵
    return world_z_depth


def create_icosphere(subdivisions, radius=1.0, noise_amplitude=0.002):
    # 黄金比例
    phi = (1 + np.sqrt(5)) / 2

    # 初始正二十面体的顶点
    vertices = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=np.float32)

    # 正规化顶点
    vertices /= np.linalg.norm(vertices, axis=1)[:, np.newaxis]

    # 初始面
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=np.int32)

    # 细分函数
    def subdivide(vertices, faces):
        new_faces = []
        for face in faces:
            v1, v2, v3 = vertices[face]
            v12 = (v1 + v2) / 2
            v23 = (v2 + v3) / 2
            v31 = (v3 + v1) / 2
            v12 /= np.linalg.norm(v12)
            v23 /= np.linalg.norm(v23)
            v31 /= np.linalg.norm(v31)
            i12 = len(vertices)
            i23 = len(vertices) + 1
            i31 = len(vertices) + 2
            vertices = np.vstack((vertices, [v12, v23, v31]))
            new_faces.extend([
                [face[0], i12, i31],
                [face[1], i23, i12],
                [face[2], i31, i23],
                [i12, i23, i31]
            ])
        return vertices, np.array(new_faces, dtype=np.int32)

    # 执行细分
    for _ in range(subdivisions):
        vertices, faces = subdivide(vertices, faces)

    # 噪声函数
    frequencies = np.random.uniform(4, 20, size=10)
    phases = np.random.uniform(0, np.pi, size=10)

    def noise_function(x, y, z):
        noise = 0
        for freq, phase in zip(frequencies, phases):
            noise += np.sin(freq * x + phase) * np.sin(freq * y + phase) * np.sin(freq * z + phase)
        return noise_amplitude * noise

    # 应用噪声
    for i in range(len(vertices)):
        x, y, z = vertices[i]
        noise = noise_function(x, y, z)
        final_radius = radius + noise
        # 沿法线方向添加噪声
        vertices[i] = final_radius * x, final_radius * y, final_radius * z

    # 重新计算法线
    normals = np.zeros_like(vertices)
    for face in faces:
        v1, v2, v3 = vertices[face]
        normal = np.cross(v2 - v1, v3 - v1)
        normals[face] += normal
    
    # 归一化法线
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]

    return vertices.flatten(), normals.flatten(), faces.flatten()


def create_sphere(rows, cols, radius=1.0):
    vertices = []
    indices = []
    
    frequencies = np.random.uniform(4, 20, size=10)
    phases = np.random.uniform(0, np.pi, size=10)

    def noise_function(x, y, z):
        noise = 0
        for freq, phase in zip(frequencies, phases):
            noise += np.sin(freq * x + phase) * np.sin(freq * y + phase) * np.sin(freq * z + phase)
        return 0.005 * noise
    
    for i in range(rows + 1):
        lat = np.pi * i / rows  # 纬度
        for j in range(cols + 1):
            lon = 2 * np.pi * j / cols  # 经度
            x = np.sin(lat) * np.cos(lon)
            y = np.cos(lat)
            z = np.sin(lat) * np.sin(lon)
            
            noise = noise_function(x, y, z)
            final_radius = radius + noise
            
            # 顶点
            vertices.extend([final_radius * x, final_radius * y, final_radius * z])
    
    for i in range(rows):
        for j in range(cols):
            first = (i * (cols + 1)) + j
            second = first + cols + 1
            
            indices.extend([first, second, first + 1])
            indices.extend([second, second + 1, first + 1])
    
    vertices = np.array(vertices, dtype=np.float32).reshape(-1, 3)
    indices = np.array(indices, dtype=np.uint32)
    
    # 计算法线
    normals = np.zeros_like(vertices)
    for i in range(0, len(indices), 3):
        v1, v2, v3 = vertices[indices[i]], vertices[indices[i+1]], vertices[indices[i+2]]
        normal = np.cross(v2 - v1, v3 - v1)
        if np.linalg.norm(normal) == 0:
            print("Warning: zero normal detected")
        normals[indices[i]] += normal
        normals[indices[i+1]] += normal
        normals[indices[i+2]] += normal
    
    # 归一化新法线
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    
    return vertices.flatten(), normals.flatten(), indices

def save_to_obj(filename, vertices, normals, indices):
    with open(filename, 'w') as f:
        # 写入顶点信息
        for i in range(0, len(vertices), 3):
            f.write(f"v {vertices[i]} {vertices[i + 1]} {vertices[i + 2]}\n")

        # 写入法线信息
        for i in range(0, len(normals), 3):
            f.write(f"vn {normals[i]} {normals[i + 1]} {normals[i + 2]}\n")

        # 写入面信息，OBJ文件的索引是从1开始的
        for i in range(0, len(indices), 3):
            f.write(f"f {indices[i] + 1}//{indices[i] + 1} {indices[i + 1] + 1}//{indices[i + 1] + 1} {indices[i + 2] + 1}//{indices[i + 2] + 1}\n")

def main():
    pygame.init()
    display = (576, 576)
    pygame.display.set_mode(display, pygame.OPENGL | pygame.DOUBLEBUF)
    
    shader = compileProgram(
        compileShader(vertex_shader, GL_VERTEX_SHADER),
        compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    )
    
    # 创建球体的顶点、法线和索引
    # rows, cols, radius = 300, 300, 0.1
    # vertices, normals, indices = create_sphere(rows, cols, radius)
    vertices, normals, indices = create_icosphere(6, 0.1)
    # save_to_obj('sphere.obj', vertices, normals, indices)
    
    # 创建 VAO 和 VBO
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)

    glBindVertexArray(VAO)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes + normals.nbytes, None, GL_STATIC_DRAW)
    glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
    glBufferSubData(GL_ARRAY_BUFFER, vertices.nbytes, normals.nbytes, normals)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    # 设置顶点属性指针
    pos_loc = glGetAttribLocation(shader, "position")
    glVertexAttribPointer(pos_loc, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(pos_loc)

    normal_loc = glGetAttribLocation(shader, "normal")
    glVertexAttribPointer(normal_loc, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(vertices.nbytes))
    glEnableVertexAttribArray(normal_loc)
    
    glUseProgram(shader)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    
    model_loc = glGetUniformLocation(shader, "model")
    view_loc = glGetUniformLocation(shader, "view")
    proj_loc = glGetUniformLocation(shader, "projection")
    
    near = 0.1
    far = 100.0
    view_position = glm.vec3(0.0, 0.0, -0.5)     # 观察者的位置
    projection = glm.perspective(glm.radians(30.0), 1.0, near, far)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, glm.value_ptr(projection))
    
    view = glm.lookAt(glm.vec3(0, 0, 0), view_position, glm.vec3(0, 1, 0))
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))
    
    # 获取 uniform 变量的位置
    ambient_loc = glGetUniformLocation(shader, "ambient_color")
    light_pos_loc = glGetUniformLocation(shader, "light_position")
    view_pos_loc = glGetUniformLocation(shader, "view_pos")
    
    # 设置 uniform 的值
    ambient_color = glm.vec4(0.05, 0.05, 0.05, 1.0)  # 柔和的环境光颜色
    light_position = glm.vec3(-0.1, -0.1, 0)     # 光源的位置

    glUniform4fv(ambient_loc, 1, glm.value_ptr(ambient_color))
    glUniform3fv(light_pos_loc, 1, glm.value_ptr(light_position))
    glUniform3fv(view_pos_loc, 1, glm.value_ptr(view_position))
    
    # 阴影部分开始  ----
    # 创建深度纹理和帧缓冲对象
    depth_texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, depth_texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, display[0], display[1], 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    borderColor = [0.0, 0.0, 0.0, 0.0]
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor)

    shadow_fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, shadow_fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_texture, 0)
    glDrawBuffer(GL_NONE)
    glReadBuffer(GL_NONE)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    #阴影部分结束  ----
    
    clock = pygame.time.Clock()
    
    model = glm.translate(glm.mat4(1.0), glm.vec3(0.0, 0.0, -0.5))
    # model = glm.rotate(model, pygame.time.get_ticks() * 0.001, glm.vec3(0.5, 1.0, 0.0))
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model))
        
    # 渲染阴影贴图开始
    light_view = glm.lookAt(light_position, glm.vec3(0.0, 0.0, -0.5), glm.vec3(0, 1, 0))
    light_projection = glm.perspective(glm.radians(30.0), 1.0, near, far)

    glBindFramebuffer(GL_FRAMEBUFFER, shadow_fbo)
    glClear(GL_DEPTH_BUFFER_BIT)
    glUseProgram(shader)
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(light_view))
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, glm.value_ptr(light_projection))
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
    
    # 调试光照
    # world_z_depth = get_world_depth_buffer(light_view, light_projection, display[0], display[1], near, far)
    # world_z_depth[world_z_depth < -1.0] = 0.0
    # world_z_depth[world_z_depth == 0.0] = world_z_depth.min()
    # world_z_depth = (world_z_depth - world_z_depth.min()) / (world_z_depth.max() - world_z_depth.min())
    # cv2.imshow("world_z_depth", (world_z_depth * 255).astype(np.uint8))
    # cv2.waitKey(1)
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    # 渲染阴影贴图结束

    # 设置回原来的渲染状态
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, glm.value_ptr(projection))
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # 正式渲染
    light_space_matrix_loc = glGetUniformLocation(shader, "light_space_matrix")
    shadow_map_loc = glGetUniformLocation(shader, "shadow_map")

    # 在渲染场景时设置阴影相关的uniform
    glUniformMatrix4fv(light_space_matrix_loc, 1, GL_TRUE, np.array(light_projection * light_view))
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, depth_texture)
    glUniform1i(shadow_map_loc, 1)
    

    # 调用函数获取世界坐标系中的Z深度矩阵
    # world_z_depth = get_world_depth_buffer(view, projection, display[0], display[1], near, far)
    # world_z_depth[world_z_depth < -1.0] = 0.0
    # world_z_depth[world_z_depth == 0.0] = world_z_depth.min()
    # world_z_depth = (world_z_depth - world_z_depth.min()) / (world_z_depth.max() - world_z_depth.min())
    # cv2.imshow("world_z_depth", (world_z_depth * 255).astype(np.uint8))
    # cv2.waitKey(1)
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()