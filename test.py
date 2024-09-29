import glfw
from OpenGL.GL import *
import numpy as np

# 初始化 GLFW
if not glfw.init():
    raise Exception("GLFW can not be initialized!")

view_size = 576

# 创建一个窗口
window = glfw.create_window(view_size, view_size, "PyOpenGL", None, None)

if not window:
    glfw.terminate()
    raise Exception("Window can not be created!")

# 设置当前的上下文
glfw.make_context_current(window)

# 顶点着色器代码
vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

# 片段着色器代码
fragment_shader = """
#version 330 core
out vec4 FragColor;

void main()
{
    FragColor = vec4(1.0f, 1.0f, 1.0f, 1.0f);
}
"""

# 编译着色器
def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)

    # 检查编译是否成功
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader))

    return shader

# 创建和链接着色器程序
def create_shader_program(vertex_src, fragment_src):
    program = glCreateProgram()

    vertex_shader = compile_shader(vertex_src, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_src, GL_FRAGMENT_SHADER)

    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)

    # 检查链接状态
    if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(program))

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return program

# 创建球体的顶点和法线
def create_sphere(rows, cols, radius=1.0):
    vertices = []
    normals = []
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
            # 法线 (由于球体，法线就是顶点位置的归一化)
            normals.extend([x, y, z])
    
    for i in range(rows):
        for j in range(cols):
            first = (i * (cols + 1)) + j
            second = first + cols + 1
            
            # 生成两个三角形用于每个四边形的表面
            indices.extend([first, second, first + 1])
            indices.extend([second, second + 1, first + 1])
    
    vertices = np.array(vertices, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)
    
    return vertices, normals, indices

def perspective(fov, aspect, near, far):
    f = 1.0 / np.tan(fov / 2.0)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ], dtype=np.float32)

def lookAt(eye, center, up):
    f = center - eye
    f = f / np.linalg.norm(f)

    u = up / np.linalg.norm(up)
    s = np.cross(f, u)
    s = s / np.linalg.norm(s)

    u = np.cross(s, f)

    result = np.eye(4, dtype=np.float32)
    result[0, :3] = s
    result[1, :3] = u
    result[2, :3] = -f
    result[0, 3] = -np.dot(s, eye)
    result[1, 3] = -np.dot(u, eye)
    result[2, 3] = np.dot(f, eye)
    
    return result

def translate(tx, ty, tz):
    translation_matrix = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    return translation_matrix

# 创建着色器程序
shader_program = create_shader_program(vertex_shader, fragment_shader)

# 创建球体的顶点、法线和索引
rows, cols, radius = 300, 300, 0.1
vertices, normals, indices = create_sphere(rows, cols, radius)

# 创建VBO（顶点缓冲区对象）和EBO（元素缓冲区对象）
VBO = glGenBuffers(1)
VAO = glGenVertexArrays(1)
EBO = glGenBuffers(1)

# 绑定VAO
glBindVertexArray(VAO)

# 绑定VBO并将顶点数据发送到GPU
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

# 绑定EBO并将索引数据发送到GPU
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

# 配置顶点属性
# 位置属性
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * vertices.itemsize, ctypes.c_void_p(0))
glEnableVertexAttribArray(0)

# 颜色属性
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * vertices.itemsize, ctypes.c_void_p(3 * vertices.itemsize))
glEnableVertexAttribArray(1)

# 解绑VAO
glBindVertexArray(0)

# 获取 uniform 变量的位置
model_loc = glGetUniformLocation(shader_program, "model")
view_loc = glGetUniformLocation(shader_program, "view")
projection_loc = glGetUniformLocation(shader_program, "projection")

# 创建透视投影矩阵
fov = np.radians(45.0)  # 视角
aspect_ratio = 1.0  # 窗口宽高比
near, far = 0.01, 100.0  # 近裁剪面和远裁剪面
projection = perspective(fov, aspect_ratio, near, far)

# 创建视图矩阵（相机位置）
camera_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
camera_target = np.array([0.0, 0.0, -0.5], dtype=np.float32)
camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
view = lookAt(camera_pos, camera_target, camera_up)

# 模型矩阵 (平移)
tx, ty, tz = 0.0, 0.0, -0.5  # 平移量
model = translate(tx, ty, tz)

# 渲染循环
while not glfw.window_should_close(window):
    # 清除屏幕
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    # 使用着色器程序
    glUseProgram(shader_program)
    
    # 传递 uniform 矩阵到着色器
    glUniformMatrix4fv(model_loc, 1, GL_TRUE, model)
    glUniformMatrix4fv(view_loc, 1, GL_TRUE, view)
    glUniformMatrix4fv(projection_loc, 1, GL_TRUE, projection)

    # 绘制球体
    glBindVertexArray(VAO)
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    # 交换缓冲区
    glfw.swap_buffers(window)

    # 处理事件
    glfw.poll_events()

# 释放资源
glDeleteVertexArrays(1, [VAO])
glDeleteBuffers(1, [VBO])
glDeleteBuffers(1, [EBO])
glDeleteProgram(shader_program)

# 终止GLFW
glfw.terminate()
