# Trabalho Prático 3

# Alunos:
# Henrique Tadashi Tarzia - 10692210
# Luis Felipe Ribeiro Chaves - 10801221

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import glm
import math
from PIL import Image

glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE);
altura = 1080
largura = 1920
window = glfw.create_window(largura, altura, "Trabalho Prático 3", None, None)
glfw.make_context_current(window)

vertex_code = """
        attribute vec3 position;
        attribute vec2 texture_coord;
        attribute vec3 normals;
       
        varying vec2 out_texture;
        varying vec3 out_fragPos;
        varying vec3 out_normal;
                
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;        
        
        void main(){
            gl_Position = projection * view * model * vec4(position,1.0);
            out_texture = vec2(texture_coord);
            out_fragPos = vec3(model * vec4(position, 1.0));
            out_normal = normals;            
        }
        """

fragment_code = """

        // parametros da iluminacao ambiente e difusa
        uniform vec3 lightPos1; // define coordenadas de posicao da luz #1
        uniform vec3 lightPos2; // define coordenadas de posicao da luz #2
        uniform float il1;  // intensidade de iluminação da luz #1
        uniform float ka;   // coeficiente de reflexao ambiente
        uniform float ia;   // intensidade de iluminação da luz ambiente
        uniform float kd;   // coeficiente de reflexao difusa
        
        // parametros da iluminacao especular
        uniform vec3 viewPos; // define coordenadas com a posicao da camera/observador
        uniform float ks; // coeficiente de reflexao especular
        uniform float ns; // expoente de reflexao especular
        
        // parametro com a cor da(s) fonte(s) de iluminacao
        vec3 lightColor = vec3(1.0, 1.0, 1.0);

        // parametros recebidos do vertex shader
        varying vec2 out_texture; // recebido do vertex shader
        varying vec3 out_normal; // recebido do vertex shader
        varying vec3 out_fragPos; // recebido do vertex shader
        uniform sampler2D samplerTexture;
        
        uniform bool invert_normal;
        
        void main(){
        
            // calculando reflexao ambiente
            vec3 ambient = ka * ia * lightColor;
        
            ////////////////////////
            // Luz #1
            ////////////////////////
            
            // calculando reflexao difusa
            vec3 norm1 = normalize(out_normal); // normaliza vetores perpendiculares
            if (invert_normal)
                norm1 = -norm1;
            vec3 lightDir1 = normalize(lightPos1 - out_fragPos); // direcao da luz
            float diff1 = max(dot(norm1, lightDir1), 0.0); // verifica limite angular (entre 0 e 90)
            vec3 diffuse1 = kd * il1 * diff1 * lightColor; // iluminacao difusa
            
            // calculando reflexao especular
            vec3 viewDir1 = normalize(viewPos - out_fragPos); // direcao do observador/camera
            vec3 reflectDir1 = reflect(-lightDir1, norm1); // direcao da reflexao
            float spec1 = pow(max(dot(viewDir1, reflectDir1), 0.0), ns);
            vec3 specular1 = ks * il1 * spec1 * lightColor;
            
            
            ////////////////////////
            // Luz #2
            ////////////////////////
            
            // calculando reflexao difusa
            vec3 norm2 = normalize(out_normal); // normaliza vetores perpendiculares
            if (invert_normal)
                norm2 = -norm2;
            vec3 lightDir2 = normalize(lightPos2 - out_fragPos); // direcao da luz
            float diff2 = max(dot(norm2, lightDir2), 0.0); // verifica limite angular (entre 0 e 90)
            vec3 diffuse2 = kd * diff2 * lightColor; // iluminacao difusa
            
            // calculando reflexao especular
            vec3 viewDir2 = normalize(viewPos - out_fragPos); // direcao do observador/camera
            vec3 reflectDir2 = reflect(-lightDir2, norm2); // direcao da reflexao
            float spec2 = pow(max(dot(viewDir2, reflectDir2), 0.0), ns);
            vec3 specular2 = ks * spec2 * lightColor;
            
            ////////////////////////
            // Combinando as duas fontes
            ////////////////////////
            
            // aplicando o modelo de iluminacao
            vec4 texture = texture2D(samplerTexture, out_texture);
            vec4 result = vec4((ambient + diffuse1 + diffuse2 + specular1 + specular2),1.0) * texture; // aplica iluminacao
            gl_FragColor = result;

        }
        """

program  = glCreateProgram()
vertex   = glCreateShader(GL_VERTEX_SHADER)
fragment = glCreateShader(GL_FRAGMENT_SHADER)

glShaderSource(vertex, vertex_code)
glShaderSource(fragment, fragment_code)

glCompileShader(vertex)
if not glGetShaderiv(vertex, GL_COMPILE_STATUS):
    error = glGetShaderInfoLog(vertex).decode()
    print(error)
    raise RuntimeError("Erro de compilacao do Vertex Shader")

glCompileShader(fragment)
if not glGetShaderiv(fragment, GL_COMPILE_STATUS):
    error = glGetShaderInfoLog(fragment).decode()
    print(error)
    raise RuntimeError("Erro de compilacao do Fragment Shader")

glAttachShader(program, vertex)
glAttachShader(program, fragment)

glLinkProgram(program)
if not glGetProgramiv(program, GL_LINK_STATUS):
    print(glGetProgramInfoLog(program))
    raise RuntimeError('Linking error')
    
glUseProgram(program)

def load_model_from_file(filename):
    """Loads a Wavefront OBJ file. """
    objects = {}
    vertices = []
    normals = []
    texture_coords = []
    faces = []

    material = None

    for line in open(filename, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'v':
            vertices.append(values[1:4])
        if values[0] == 'vn':
            normals.append(values[1:4])
        elif values[0] == 'vt':
            texture_coords.append(values[1:3])
        elif values[0] in ('usemtl', 'usemat'):
            material = values[1]
        elif values[0] == 'f':
            face = []
            face_texture = []
            face_normals = []
            for v in values[1:]:
                w = v.split('/')
                face.append(int(w[0]))
                face_normals.append(int(w[2]))
                if len(w) >= 2 and len(w[1]) > 0:
                    face_texture.append(int(w[1]))
                else:
                    face_texture.append(0)

            faces.append((face, face_texture, face_normals, material))

    model = {}
    model['vertices'] = vertices
    model['texture'] = texture_coords
    model['faces'] = faces
    model['normals'] = normals

    return model

glEnable(GL_TEXTURE_2D)
qtd_texturas = 10
textures = glGenTextures(qtd_texturas)

def load_texture_from_file(texture_id, img_textura):
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    img = Image.open(img_textura)
    img_width = img.size[0]
    img_height = img.size[1]
    image_data = img.tobytes("raw", "RGB", 0, -1)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img_width, img_height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)

id_textura = 0
vertices_list = []    
normals_list = []    
textures_coord_list = []

def insert_model(file_name,texture_name):
    global id_textura, vertices_list, textures_coord_list

    object_id = id_textura
    id_textura += 1

    modelo = load_model_from_file(file_name)

    vi = len(vertices_list)   

    print('Processando modelo ' + str(file_name) + ' ...')
    print('Vértice inicial: ' + str(len(vertices_list)))

    for face in modelo['faces']:
        for vertice_id in face[0]:
            vertices_list.append( modelo['vertices'][vertice_id-1] )
        for texture_id in face[1]:
            textures_coord_list.append( modelo['texture'][texture_id-1] )
        for normal_id in face[2]:
            normals_list.append( modelo['normals'][normal_id-1] )  

    print('Vértice final: ' + str(len(vertices_list)))

    vf = len(vertices_list)        

    load_texture_from_file(object_id,texture_name)

    return vi,(vf-vi),object_id

#-----------SKY------------#

angle_sky = 0.0
r_sky = (0.0, 1.0, 0.0)
s_sky = (1.0, 1.0, 1.0)
t_sky = (-s_sky[0] / 2, -s_sky[1] / 2, -s_sky[2] / 2)
vi_sky, qtd_sky, id_sky = insert_model('modelos/sky/cubo.obj','modelos/sky/sky_modified.png')

#---------TERRAIN (GRASS)----------#

angle_grass = 0.0
r_grass = (0.0, 1.0, 0.0)
t_grass = (0.0, -1.0, -12.0)
s_grass = (27.0, 27.0, 27.0)
vi_grass, qtd_grass, id_grass = insert_model('modelos/terrain/terrain.obj','modelos/terrain/grass.jpg')

#---------TERRAIN (STREET)----------#

angle_street = 0.0
r_street = (0.0, 1.0, 0.0)
t_street = (0.0, -1.0, 20.0)
s_street = (28.0, 1.0, 5.0)
vi_street, qtd_street, id_street = insert_model('modelos/terrain/terrain-02.obj','modelos/terrain/street.jpg')

#-------------BIRDBATH-------------#

angle_birdbath = 0.0
r_birdbath = (0.0, 1.0, 0.0)
t_birdbath = (8.0, -1.0, 0.0)
s_birdbath = (2.0,2.0,2.0)
vi_birdbath, qtd_birdbath, id_birdbath = insert_model('modelos/birdbath/birdbath.obj','modelos/birdbath/birdbath.jpg')

#-------------MUSEUM-------------#

angle_museum = 0.0
r_museum = (0.0, 1.0, 0.0)
t_museum = (-5.0, 0.7, 0.0)
s_museum = (0.1, 0.1, 0.1)
vi_museum, qtd_museum, id_museum = insert_model('modelos/museum/museum.obj','modelos/museum/museum.jpeg')

#-------FLOOR (MUSEUM)-------#

angle_floor_m = 0.0
r_floor_m = (0.0, 1.0, 0.0)
t_floor_m = (-5.0, -0.55, 0.0)
s_floor_m = (6.25,1,3.9)
vi_floor_m, qtd_floor_m, id_floor_m = insert_model('modelos/floor/floor.obj','modelos/floor/brick_hexagonal.png')

#------------TABLE-----------#

angle_table = 0.0
r_table = (0.0, 0.0, 1.0)
t_table = (0.8,-0.55,0.5)
s_table = (1.0, 1.0, 1.0)
vi_table, qtd_table, id_table = insert_model('modelos/table/table.obj','modelos/table/table.png')

#------------DUCK------------#

angle_duck = 0.0
r_duck = (0.0, 1.0, 0.0)
t_duck = (-5.0,0.1, 0.0)
s_duck = (0.28,0.28,0.28)
vi_duck, qtd_duck, id_duck = insert_model('modelos/duck/duck.obj','modelos/duck/duck.png')

#-----------PAINT #1----------#

angle_paint_1 = 0.0
r_paint_1 = (0.0,1.0,0.0)
t_paint_1 = (-2.0,1.5,-3.7)
s_paint_1 = (0.2,0.2,0.2)     
vi_paint_1, qtd_paint_1, id_paint_1 = insert_model('modelos/paints/paint-01.obj','modelos/paints/paint-01.jpg')

#-----------PAINT #2----------#

angle_paint_2 = 0.0
r_paint_2 = (0.0,1.0,0.0)
t_paint_2 = (-0.5,1.0,-3.7)
s_paint_2 = (0.15,0.15,0.15)
vi_paint_2, qtd_paint_2, id_paint_2 = insert_model('modelos/paints/paint-02.obj','modelos/paints/paint-02.jpg')

#-------------SEEDS-------------#

angle_seeds = 0.0
r_seeds = (0.0,1.0, 0.0)
t_seeds = (4.0,-1.0,-2.0)
s_seeds = (12.0,12.0,12.0)
vi_seeds, qtd_seeds, id_seeds = insert_model('modelos/seeds/seeds.obj','modelos/seeds/seeds.png')

#-------------APPLES-------------#

angle_apple = (10.0,0.0,15.0)
r_apple = ([1.0,0.0,1.0],[0.0,0.0,1.0],[0.0,0.0,1.0])                                                                 
t_apple = ([3.6,-1.4,0.2],[4.2,-1.0,-0.2],[3.6,-2.0,-0.5])
s_apple = ([4.0,4.0,4.0],[4.0,4.0,4.0],[4.0,4.0,4.0])
vi_apple, qtd_apple, id_apple = insert_model('modelos/apple/apple.obj','modelos/apple/apple.png') 

#-------------CAMPFIRE-------------#

angle_campfire = 0.0
r_campfire = (0.0,1.0,0.0)
t_campfire = (8.0,-1.0,7.0)
s_campfire = (0.02,0.02,0.02)                                                                                                                                                                                      # expoente de reflexao especular
vi_campfire, qtd_campfire, id_campfire = insert_model('modelos/campfire/campfire.obj','modelos/campfire/campfire.png') 

#------MATERNIDADE SCULPTURE------#

angle_maternidade = 0.0
r_maternidade = (0.0,1.0,0.0)
t_maternidade = (-8.0,-0.7,-2.6)
s_maternidade = (1.0,1.0,1.0)
vi_maternidade, qtd_maternidade, id_maternidade = insert_model('modelos/maternidade_sculpture/maternidade_modified.obj','modelos/maternidade_sculpture/maternidade.jpg')

#-------ALLEGORIE SCULPTURE-------#

angle_allegorie = 0.0
r_allegorie = (0.0,1.0,0.0)
t_allegorie = (0.8,-1.0,-1.5)
s_allegorie = (0.1,0.1,0.1)
vi_allegorie, qtd_allegorie, id_allegorie = insert_model('modelos/allegorie_sculpture/Allegorie_modifiedobj.obj','modelos/allegorie_sculpture/allegorie.png')

#-------------HACHIROKU-------------#

angle_car = 0.0
r_car = (0.0, 1.0, 0.0)
t_car = (-16.0, -1.0, 16.2)
s_car = (1.2,1.2,1.2)
vi_car, qtd_car, id_car = insert_model('modelos/hachiroku/hachiroku.obj','modelos/hachiroku/hachiroku.jpg')

#-------------SOCCER BALL-------------#

angle_ball = 0.0
r_ball = (0.0, 0.0, 1.0)
t_ball = (-15.0, -0.75, 5.0)
s_ball = (0.03,0.03,0.03)
vi_ball, qtd_ball, id_ball = insert_model('modelos/soccer_ball/soccer_ball.obj','modelos/soccer_ball/soccer_ball.jpg')

#-------------BOOTH-------------#

angle_booth = 0.0
r_booth = (0.0, 0.0, 1.0)
t_booth = (-16.0, -1.0, 12.0)
s_booth = (2.0,2.0,2.0)
vi_booth, qtd_booth, id_booth = insert_model('modelos/booth/booth.obj','modelos/booth/booth.jpg')

#-------------HYDRANT------------#

angle_hydrant = 0.0
r_hydrant = (0.0, 0.0, 1.0)
t_hydrant = (-10.0, -1.0, 13.0)
s_hydrant = (0.01,0.01,0.01)
vi_hydrant, qtd_hydrant, id_hydrant = insert_model('modelos/hydrant/hydrant.obj','modelos/hydrant/hydrant.jpg')

#-------------EAGLE-------------#

angle_eagle = 0.0
r_eagle = (0.0,1.0,0.0)
t_eagle = (3.0,7.0,0.0)
s_eagle = (1.0,1.0,1.0)
vi_eagle, qtd_eagle, id_eagle = insert_model('modelos/eagle/eagle.obj','modelos/eagle/eagle.png')

#-------------OAK---------------#

angle_oak = 0.0
r_oak = (0.0,1.0,0.0)
t_oak = (6.0, -1.0, -4.0)
s_oak = (0.06,0.06,0.06)
vi_oak, qtd_oak, id_oak = insert_model('modelos/tree/oak.obj','modelos/tree/oak.jpg')

#----------POKEBALL---------#

angle_pokeball = -90.0
r_pokeball = (0.0, 1.0, 0.0)
t_pokeball = (0.8,0.05,-0.6)
s_pokeball = (0.03,0.03,0.03)
vi_pokeball, qtd_pokeball, id_pokeball = insert_model('modelos/table/pokeball.obj','modelos/table/pokeball.png')

#-------------SUN-------------#

angle_sun = 0.0
r_sun = (0.0,0.0,1.0)
t_sun = (14.0, 14.0, 0.0)
s_sun = (1.0,1.0,1.0)
vi_sun, qtd_sun, id_sun = insert_model('modelos/light/sun_modified.obj','modelos/light/sun.jpg') 

#-------------END----------------#

buffer = glGenBuffers(3)

vertices = np.zeros(len(vertices_list), [("position", np.float32, 3)])
vertices['position'] = vertices_list


glBindBuffer(GL_ARRAY_BUFFER, buffer[0])
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
stride = vertices.strides[0]
offset = ctypes.c_void_p(0)
loc_vertices = glGetAttribLocation(program, "position")
glEnableVertexAttribArray(loc_vertices)
glVertexAttribPointer(loc_vertices, 3, GL_FLOAT, False, stride, offset)

textures = np.zeros(len(textures_coord_list), [("position", np.float32, 2)]) # duas coordenadas
textures['position'] = textures_coord_list


glBindBuffer(GL_ARRAY_BUFFER, buffer[1])
glBufferData(GL_ARRAY_BUFFER, textures.nbytes, textures, GL_STATIC_DRAW)
stride = textures.strides[0]
offset = ctypes.c_void_p(0)
loc_texture_coord = glGetAttribLocation(program, "texture_coord")
glEnableVertexAttribArray(loc_texture_coord)
glVertexAttribPointer(loc_texture_coord, 2, GL_FLOAT, False, stride, offset)

normals = np.zeros(len(normals_list), [("position", np.float32, 3)]) # três coordenadas
normals['position'] = normals_list

glBindBuffer(GL_ARRAY_BUFFER, buffer[2])
glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
stride = normals.strides[0]
offset = ctypes.c_void_p(0)
loc_normals_coord = glGetAttribLocation(program, "normals")
glEnableVertexAttribArray(loc_normals_coord)
glVertexAttribPointer(loc_normals_coord, 3, GL_FLOAT, False, stride, offset)

def desenha_objeto(vi, qtd, texture_id, angle, rotation, translation, scale, ka = 1, kd = 1, ks = 1, ns = 4096, invert_normal = False):
    mat_model = model(angle, *rotation, *translation, *scale)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    
    loc_ka = glGetUniformLocation(program, "ka")
    glUniform1f(loc_ka, ka)
    loc_kd = glGetUniformLocation(program, "kd")
    glUniform1f(loc_kd, kd)   
    loc_ks = glGetUniformLocation(program, "ks")
    glUniform1f(loc_ks, ks)
    loc_ns = glGetUniformLocation(program, "ns")
    glUniform1f(loc_ns, ns)
    loc_invert_normal = glGetUniformLocation(program, "invert_normal")
    glUniform1f(loc_invert_normal, invert_normal)

    glBindTexture(GL_TEXTURE_2D, texture_id)
    glDrawArrays(GL_TRIANGLES, vi, qtd)

def desenha_objeto_rotacao(vi, qtd, texture_id, angle, rotation, translation, scale, ka = 1, kd = 1, ks = 1, ns = 4096, invert_normal = False):

    global vertices

    angle = math.radians(angle)

    # aplica a matriz model
    mat_model = glm.mat4(1.0)                                                                # instanciando uma matriz identidade
    mat_model = glm.translate(mat_model, glm.vec3(-translation[0], 0.0, -translation[2]))    # aplicando translacao para a origem
    mat_model = glm.rotate(mat_model, angle, glm.vec3(*rotation))                            # aplicando rotacao 
    mat_model = glm.translate(mat_model, glm.vec3(*translation))                             # aplicando translacao de volta ao ponto original
       
    mat_model = glm.scale(mat_model, glm.vec3(*scale))                  # aplicando escala
    mat_model = np.array(mat_model).T                                   # pegando a transposta da matriz (glm trabalha com ela invertida)

    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model) 

    loc_ka = glGetUniformLocation(program, "ka")
    glUniform1f(loc_ka, ka)                      
    
    loc_kd = glGetUniformLocation(program, "kd")
    glUniform1f(loc_kd, kd)    

    loc_ks = glGetUniformLocation(program, "ks")
    glUniform1f(loc_ks, ks)       
    
    loc_ns = glGetUniformLocation(program, "ns")
    glUniform1f(loc_ns, ns)

    # define o id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, texture_id)

    # desenha o objeto 
    glDrawArrays(GL_TRIANGLES, vi, qtd)


def desenha_luz1(vi, qtd, texture_id, angle, rotation, translation, scale, ka = 0.5, kd = 1, ks = 0, ns = 1, invert_normal = True):
    mat_model = model(angle, *rotation, *translation, *scale)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    
    loc_ka = glGetUniformLocation(program, "ka")
    glUniform1f(loc_ka, ka)
    loc_kd = glGetUniformLocation(program, "kd")
    glUniform1f(loc_kd, kd)
    loc_ks = glGetUniformLocation(program, "ks")
    glUniform1f(loc_ks, ks)
    loc_ns = glGetUniformLocation(program, "ns")
    glUniform1f(loc_ns, ns)
    loc_invert_normal = glGetUniformLocation(program, "invert_normal")
    glUniform1f(loc_invert_normal, invert_normal)

    loc_light_pos = glGetUniformLocation(program, "lightPos1")
    glUniform3f(loc_light_pos, *translation)

    glBindTexture(GL_TEXTURE_2D, texture_id)
    glDrawArrays(GL_TRIANGLES, vi, qtd)
    
def desenha_luz2(vi, qtd, texture_id, angle, rotation, translation, scale, ka = 1, kd = 0, ks = 0, ns = 1, invert_normal = True):
    mat_model = model(angle, *rotation, *translation, *scale)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    
    loc_ka = glGetUniformLocation(program, "ka")
    glUniform1f(loc_ka, ka)
    loc_kd = glGetUniformLocation(program, "kd")
    glUniform1f(loc_kd, kd)
    loc_ks = glGetUniformLocation(program, "ks")
    glUniform1f(loc_ks, ks)
    loc_ns = glGetUniformLocation(program, "ns")
    glUniform1f(loc_ns, ns)
    loc_invert_normal = glGetUniformLocation(program, "invert_normal")
    glUniform1f(loc_invert_normal, invert_normal)

    nova_coordenada_luz = glm.mat4(1.0)
    nova_coordenada_luz = glm.rotate(nova_coordenada_luz, math.radians(angle), glm.vec3(*rotation))
    nova_coordenada_luz = nova_coordenada_luz * glm.vec4(*translation, 1.0)

    loc_light_pos = glGetUniformLocation(program, "lightPos2")
    glUniform3f(loc_light_pos, nova_coordenada_luz[0], nova_coordenada_luz[1], nova_coordenada_luz[2])

    glBindTexture(GL_TEXTURE_2D, texture_id)
    glDrawArrays(GL_TRIANGLES, vi, qtd)

cameraPos   = glm.vec3(0.0,  1.0,  0.0);
cameraFront = glm.vec3(0.0,  0.0, -1.0);
cameraUp    = glm.vec3(0.0,  1.0,  0.0);

polygonal_mode = False
internalLight = True
ia = 1.0
loc_ia = glGetUniformLocation(program, "ia")
glUniform1f(loc_ia, 1)

def key_event(window,key,scancode,action,mods):
    global cameraPos, cameraFront, cameraUp, polygonal_mode, internalLight, ia
    
    cameraSpeed = 0.05
    if key == 87 and (action==1 or action==2): # tecla W
        cameraPos += cameraSpeed * cameraFront
    
    if key == 83 and (action==1 or action==2): # tecla S
        cameraPos -= cameraSpeed * cameraFront
    
    if key == 65 and (action==1 or action==2): # tecla A
        cameraPos -= glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed
        
    if key == 68 and (action==1 or action==2): # tecla D
        cameraPos += glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed
        
    if key == 79 and action==1 and polygonal_mode==True:
        polygonal_mode=False
    else:
        if key == 79 and action==1 and polygonal_mode==False:
            polygonal_mode=True
    
    if key == 76 and (action==1 or action==2): # tecla L
        internalLight = not internalLight

    if key == 80 and (action==1 or action==2) and ia < 1.0: # tecla P
        ia += 0.1
        loc_ia = glGetUniformLocation(program, "ia")
        glUniform1f(loc_ia, ia)

    if key == 85 and (action==1 or action==2) and ia > 0.0: # tecla U
        ia -= 0.1
        loc_ia = glGetUniformLocation(program, "ia")
        glUniform1f(loc_ia, ia)
        
        
firstMouse = True
yaw = -90.0 
pitch = 0.0
lastX =  largura/2
lastY =  altura/2

def mouse_event(window, xpos, ypos):
    global firstMouse, cameraFront, yaw, pitch, lastX, lastY
    if firstMouse:
        lastX = xpos
        lastY = ypos
        firstMouse = False

    xoffset = xpos - lastX
    yoffset = lastY - ypos
    lastX = xpos
    lastY = ypos

    sensitivity = 0.3 
    xoffset *= sensitivity
    yoffset *= sensitivity

    yaw += xoffset;
    pitch += yoffset;

    if pitch >= 90.0: pitch = 90.0
    if pitch <= -90.0: pitch = -90.0

    front = glm.vec3()
    front.x = math.cos(glm.radians(yaw)) * math.cos(glm.radians(pitch))
    front.y = math.sin(glm.radians(pitch))
    front.z = math.sin(glm.radians(yaw)) * math.cos(glm.radians(pitch))
    cameraFront = glm.normalize(front)
    
glfw.set_key_callback(window,key_event)
glfw.set_cursor_pos_callback(window, mouse_event)

def model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z):
    
    angle = math.radians(angle)
    
    matrix_transform = glm.mat4(1.0)                                                # instanciando uma matriz identidade
    matrix_transform = glm.rotate(matrix_transform, angle, glm.vec3(r_x, r_y, r_z)) # aplicando rotacao
    matrix_transform = glm.translate(matrix_transform, glm.vec3(t_x, t_y, t_z))     # aplicando translacao
    matrix_transform = glm.scale(matrix_transform, glm.vec3(s_x, s_y, s_z))         # aplicando escala
    
    matrix_transform = np.array(matrix_transform).T # pegando a transposta da matriz (glm trabalha com ela invertida)
    
    return matrix_transform

def view():
    global cameraPos, cameraFront, cameraUp
    mat_view = glm.lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    mat_view = np.array(mat_view)
    return mat_view

def projection():
    global altura, largura
    # perspective parameters: fovy, aspect, near, far
    mat_projection = glm.perspective(glm.radians(45.0), largura/altura, 0.1, 1000.0)
    mat_projection = np.array(mat_projection)    
    return mat_projection

glfw.show_window(window)
glfw.set_cursor_pos(window, lastX, lastY)

glEnable(GL_DEPTH_TEST) 

# PROPRIEDADES DE ILUMINAÇÃO DOS OBJETOS

#---------------MUSEUM-------------#
ka_museum = 0.7
kd_museum = 0.5
ks_museum = 0.0
ns_museum = 1.0

#----------FLOOR (MUSEUM)----------#
ka_floor_m = 0.6
kd_floor_m = 0.6
ks_floor_m = 0.3
ns_floor_m = 4.0

#---------------TABLE--------------#
ka_table = 0.7
kd_table = 0.5
ks_table = 0.0
ns_table = 1.0

#---------------DUCK---------------#
ka_duck = 0.8
kd_duck = 0.5
ks_duck = 0.0
ns_duck = 1.0

#------MATERNIDADE SCULPTURE-------#
ka_maternidade = 1.0
kd_maternidade = 0.5
ks_maternidade = 1.0
ns_maternidade = 256.0

#-------ALLEGORIE SCULPTURE--------#
ka_allegorie = 1.0
kd_allegorie = 0.5
ks_allegorie = 1.0
ns_allegorie = 256.0

#--------------PAINT #1------------#
ka_paint1 = 0.8
kd_paint1 = 0.4
ks_paint1 = 0.1
ns_paint1 = 8.0

#-------------PAINT #2-------------#
ka_paint2 = 0.8
kd_paint2 = 0.4
ks_paint2 = 0.1
ns_paint2 = 8.0

#-----------FLOOR (GRASS)----------#
ka_grass = 1.0
kd_grass = 0.8
ks_grass = 0.4
ns_grass = 32.0

#-----------FLOOR (STREET)---------#
ka_street = 0.8
kd_street = 1.0
ks_street = 0.0
ns_street = 16.0

#---------------EAGLE--------------#
ka_eagle = 0.8
kd_eagle = 0.9
ks_eagle = 0.0
ns_eagle = 1.0

#-------------BIRDBATH-------------#
ka_birdbath = 1.0
kd_birdbath = 0.6
ks_birdbath = 0.4
ns_birdbath = 16.0

#--------------CAMPFIRE------------#
ka_campfire = 0.8
kd_campfire = 1.0
ks_campfire = 0.0
ns_campfire = 1.0

#---------------SEEDS--------------#
ka_seeds = 1.0
kd_seeds = 1.0
ks_seeds = 0.3
ns_seeds = 32.0

#---------------APPLE--------------#
ka_apple = 1.0
kd_apple = 0.7
ks_apple = 0.8
ns_apple = 64.0

#----------------OAK---------------#
ka_oak = 1.0
kd_oak = 0.6
ks_oak = 0.0
ns_oak = 1.0

#---------------BOOTH--------------#
ka_booth = 1.0
kd_booth = 0.7
ks_booth = 1.0
ns_booth = 128.0

#------------SOCCER BALL------------#
ka_ball = 0.6
kd_ball = 0.6
ks_ball = 0.3
ns_ball = 128.0

#--------------HYDRANT-------------#
ka_hydrant = 1.0
kd_hydrant = 0.5
ks_hydrant = 1.0
ns_hydrant = 64.0

#-------------HACHIROKU------------#
ka_car = 1.0
kd_car = 0.6
ks_car = 1.0
ns_car = 128.0
    
while not glfw.window_should_close(window):

    glfw.poll_events() 
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    glClearColor(0.2, 0.2, 0.2, 1.0)
    
    if polygonal_mode==True:
        glPolygonMode(GL_FRONT_AND_BACK,GL_LINE)
    if polygonal_mode==False:
        glPolygonMode(GL_FRONT_AND_BACK,GL_FILL)

    # desenha_objeto(vi_sky, qtd_sky, id_sky, angle_sky, r_sky, t_sky, s_sky)                                                       # RUIM
    
    # ÁREA INTERNA
    desenha_objeto(vi_museum, qtd_museum, id_museum, angle_museum, r_museum, t_museum, s_museum, ka_museum, kd_museum, ks_museum, ns_museum)                                                       # MUSEUM
    desenha_objeto(vi_floor_m, qtd_floor_m, id_floor_m, angle_floor_m, r_floor_m, t_floor_m, s_floor_m, ka_floor_m, kd_floor_m, ks_floor_m, ns_floor_m)                                            # FLOOR (MUSEUM)
    desenha_objeto(vi_table, qtd_table, id_table, angle_table, r_table, t_table, s_table, ka_table, kd_table, ks_table, ns_table)                                                                  # TABLE
    desenha_objeto(vi_paint_1, qtd_paint_1, id_paint_1, angle_paint_1, r_paint_1, t_paint_1, s_paint_1, ka_paint1, kd_paint1, ks_paint1, ns_paint1)                                                # PAINT #1
    desenha_objeto(vi_paint_2, qtd_paint_2, id_paint_2, angle_paint_2, r_paint_2, t_paint_2, s_paint_2, ka_paint2, kd_paint2, ks_paint2, ns_paint2)                                                # PAINT #2
    desenha_objeto(vi_allegorie, qtd_allegorie, id_allegorie, angle_allegorie, r_allegorie, t_allegorie, s_allegorie, ka_allegorie, kd_allegorie, ks_allegorie, ns_allegorie)                      # ALLEGOIRE
    desenha_objeto(vi_maternidade, qtd_maternidade, id_maternidade, angle_maternidade, r_maternidade, t_maternidade, s_maternidade, ka_maternidade,kd_maternidade, ks_maternidade, ns_maternidade) # MATERNIDADE
    desenha_objeto(vi_duck, qtd_duck, id_duck, angle_duck, r_duck, t_duck, s_duck, ka_duck, kd_duck, ks_duck, ns_duck)                                                                             # DUCK

    # ÁREA EXTERNA
    desenha_objeto_rotacao(vi_eagle, qtd_eagle, id_eagle, angle_eagle, r_eagle, t_eagle, s_eagle, ka_eagle, kd_eagle, ks_eagle, ns_eagle)                           # EAGLE
    desenha_objeto(vi_birdbath, qtd_birdbath, id_birdbath, angle_birdbath, r_birdbath, t_birdbath, s_birdbath, ka_birdbath, kd_birdbath, ks_birdbath, ns_birdbath)  # BIRDBATH
    desenha_objeto(vi_grass, qtd_grass, id_grass, angle_grass, r_grass, t_grass, s_grass, ka_grass, kd_grass, ks_grass, ns_grass)                                   # FLOOR (GRASS)
    desenha_objeto(vi_street, qtd_street, id_street, angle_street, r_street, t_street, s_street,ka_street, kd_street, ks_street,ns_street)                          # FLOOR (STREET)
    desenha_objeto(vi_campfire, qtd_campfire, id_campfire, angle_campfire, r_campfire, t_campfire, s_campfire, ka_campfire, kd_campfire, ks_campfire, ns_campfire)  # CAMPFIRE
    desenha_objeto(vi_seeds, qtd_seeds, id_seeds, angle_seeds, r_seeds, t_seeds, s_seeds, ka_seeds, kd_seeds, ks_seeds, ns_seeds)                                   # SEEDS
    desenha_objeto(vi_apple, qtd_apple, id_apple, angle_apple[0], r_apple[0], t_apple[0], s_apple[0], ka_apple, kd_apple, ks_apple, ns_apple)                       # APPLE #1
    desenha_objeto(vi_apple, qtd_apple, id_apple, angle_apple[1], r_apple[1], t_apple[1], s_apple[1], ka_apple, kd_apple, ks_apple, ns_apple)                       # APPLE #2
    desenha_objeto(vi_apple, qtd_apple, id_apple, angle_apple[2], r_apple[2], t_apple[2], s_apple[2], ka_apple, kd_apple, ks_apple, ns_apple)                       # APPLE #3
    desenha_objeto(vi_oak, qtd_oak, id_oak, angle_oak, r_oak, t_oak, s_oak, ka_oak, kd_oak, ks_oak, ns_oak)                                                         # OAK
    desenha_objeto(vi_booth, qtd_booth, id_booth, angle_booth, r_booth, t_booth, s_booth, ka_booth, kd_booth, ks_booth, ns_booth)                                   # BOOTH
    desenha_objeto(vi_hydrant, qtd_hydrant, id_hydrant, angle_hydrant, r_hydrant, t_hydrant, s_hydrant, ka_hydrant, kd_hydrant, ks_hydrant, ns_hydrant)             # HYDRANT   
    desenha_objeto(vi_car, qtd_car, id_car, angle_car, r_car, t_car, s_car, ka_car, kd_car, ks_car, ns_car)                                                         # HACHIROKU
    desenha_objeto(vi_ball, qtd_ball, id_ball, angle_ball, r_ball, t_ball, s_ball, ka_ball, kd_ball, ks_ball, ns_ball)                                              # SOCCER BALL

    loc_il1 = glGetUniformLocation(program, "il1")
    glUniform1f(loc_il1, int(internalLight))
    desenha_luz1(vi_pokeball, qtd_pokeball, id_pokeball, angle_pokeball, r_pokeball, t_pokeball, s_pokeball)

    desenha_luz2(vi_sun, qtd_sun, id_sun, angle_sun, r_sun, t_sun, s_sun)

    angle_sun += 1.0
    angle_eagle -= 1.0

    if angle_eagle == -360.0: angle_eagle = 0.0
    
    mat_view = view()
    loc_view = glGetUniformLocation(program, "view")
    glUniformMatrix4fv(loc_view, 1, GL_FALSE, mat_view)

    mat_projection = projection()
    loc_projection = glGetUniformLocation(program, "projection")
    glUniformMatrix4fv(loc_projection, 1, GL_FALSE, mat_projection)    
    
    loc_view_pos = glGetUniformLocation(program, "viewPos")
    glUniform3f(loc_view_pos, cameraPos[0], cameraPos[1], cameraPos[2])
    
    glfw.swap_buffers(window)

glfw.terminate()

