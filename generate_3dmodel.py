import numpy as np
from vispy import app, gloo
from vispy.util.transforms import perspective, translate, rotate

# 顶点着色器
vertex_shader = """
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
attribute vec3 position;
attribute vec3 normal;
varying vec3 v_normal;
varying vec3 frag_pos;
void main()
{
    v_normal = normal;
    frag_pos = vec3(model * vec4(position, 1.0));
    gl_Position = projection * view * model * vec4(position, 1.0);
}
"""

# 片段着色器
fragment_shader = """
varying vec3 v_normal;
varying vec3 frag_pos;
uniform vec4 ambient_color;
uniform vec3 light_position;
void main()
{
    // Normalize the normal and calculate light direction from light position
    vec3 norm = normalize(v_normal);
    vec3 light_dir = normalize(light_position - frag_pos);
    vec3 stripe_dir = normalize(frag_pos - light_position);
    float stripe_pattern = (sin(dot(stripe_dir, vec3(0.0, 1.0, 0.0)) * 200) + 1.0) / 2.0;
    stripe_pattern = pow(stripe_pattern, 2.0);
    vec4 stripe_color = vec4(stripe_pattern, stripe_pattern, stripe_pattern, 1.0);
    
    // Diffuse lighting calculation
    float diff = max(dot(norm, light_dir), 0.0);

    vec4 diffuse_color = vec4(diff, diff, diff, 1.0);
    vec4 color = ambient_color + diffuse_color * 0.5 + stripe_color * 0.5;
    
    gl_FragColor = color;
}
"""

# 创建球体的顶点和法线，增加 radius 和噪声扰动
def create_sphere(rows, cols, radius=1.0, bumpiness=0.005):
    vertices = []
    normals = []
    indices = []

    # 设置一个噪声扰动函数，可以使用正弦函数或者Perlin噪声
    def noise_function(x, y, z):
        # 简单的正弦波扰动
        noise = np.sin(10 * x) * np.sin(10 * y) * np.sin(10 * z)  # 频率调节
        return noise

    for i in range(rows):
        theta = np.pi * i / (rows - 1)  # 纬度
        for j in range(cols):
            phi = 2 * np.pi * j / (cols - 1)  # 经度
            x = np.sin(theta) * np.cos(phi)
            y = np.cos(theta)
            z = np.sin(theta) * np.sin(phi)

            # 增加扰动的半径
            noise = noise_function(x, y, z)
            final_radius = radius + bumpiness * noise

            # 应用扰动后的顶点
            vertices.append([final_radius * x, final_radius * y, final_radius * z])
            normals.append([x, y, z])  # 法线依旧使用单位向量

    vertices = np.array(vertices, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)

    for i in range(rows - 1):
        for j in range(cols - 1):
            indices.append([i * cols + j, i * cols + (j + 1), (i + 1) * cols + j])
            indices.append([(i + 1) * cols + j, i * cols + (j + 1), (i + 1) * cols + (j + 1)])

    indices = np.array(indices, dtype=np.uint32)
    return vertices, normals, indices

# 设置行数、列数、半径和球体位置来控制球体的分辨率、大小和位置
rows, cols, radius = 300, 300, 0.1
vertices, normals, indices = create_sphere(rows, cols, radius)

# 创建程序对象
program = gloo.Program(vertex_shader, fragment_shader)

# 绑定顶点数据
program['position'] = gloo.VertexBuffer(vertices)
program['normal'] = gloo.VertexBuffer(normals)
program['ambient_color'] = (0.1, 0.1, 0.1, 1.0)  # 环境光的颜色和强度
program['light_position'] = (0, -0.1, 0)  # 设置点光源的位置

# 定义索引缓冲区对象 (IBO)
ibo = gloo.IndexBuffer(indices)

def lookAt(eye, target, up):
    """生成一个lookAt矩阵"""
    f = np.array(target) - np.array(eye)
    f = f / np.linalg.norm(f)
    
    r = np.cross(f, np.array(up))
    r = r / np.linalg.norm(r)
    
    u = np.cross(r, f)

    lookat_matrix = np.array([
        [r[0], r[1], r[2], -np.dot(r, eye)],
        [u[0], u[1], u[2], -np.dot(u, eye)],
        [-f[0], -f[1], -f[2], np.dot(f, eye)],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    
    return lookat_matrix

# 创建场景画布
class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, keys='interactive', size=(600, 600))
        self.theta = 0
        self.phi = 0

        gloo.set_clear_color('black')
        gloo.set_state('opaque')

        # 默认的模型矩阵、视图矩阵和投影矩阵
        self.model = np.eye(4, dtype=np.float32)
        
        # 假设相机位于 (2, 2, 2)，目标点是 (0, 0, 0)，上方向为 (0, 1, 0)
        eye = [0.0, 0.0, 0.0]
        target = [0.0, 0.0, -0.5]
        up = [0.0, 1.0, 0.0]
        
        # 生成视图矩阵
        self.view = lookAt(eye, target, up)

        self.projection = perspective(45.0, 1.0, 0.1, 10.0)

        # 设置初始位置，使用 translate 函数进行位置的平移
        self.sphere_position = translate((0.0, 0.0, -0.4))  # 球体的初始位置
        program['model'] = self.model
        program['view'] = self.view
        program['projection'] = self.projection

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.physical_size)

    def on_draw(self, event):
        gloo.clear()

        # 旋转球体
        self.theta += 1
        self.phi += 1

        # 应用旋转和位置平移
        rotation = np.dot(rotate(self.theta, (0, 1, 0)), rotate(self.phi, (1, 0, 0)))
        model = np.dot(self.sphere_position, rotation)  # 位置平移 + 旋转
        program['model'] = model

        # 绘制球体
        program.draw('triangles', ibo)

    # 支持通过键盘输入来移动球体的位置
    def on_key_press(self, event):
        if event.text == 'w':
            self.sphere_position = np.dot(translate((0.0, 0.1, 0.0)), self.sphere_position)  # 向上移动
        elif event.text == 's':
            self.sphere_position = np.dot(translate((0.0, -0.1, 0.0)), self.sphere_position)  # 向下移动
        elif event.text == 'a':
            self.sphere_position = np.dot(translate((-0.1, 0.0, 0.0)), self.sphere_position)  # 向左移动
        elif event.text == 'd':
            self.sphere_position = np.dot(translate((0.1, 0.0, 0.0)), self.sphere_position)  # 向右移动

# 创建并运行应用程序
canvas = Canvas()
canvas.show()
app.run()
