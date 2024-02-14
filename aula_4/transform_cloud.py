import open3d as o3d
import numpy as np
import os
from scipy.spatial.transform import Rotation
import copy


def angulos_euler_para_matriz_rotacao(angulos_euler):
    rotacao = Rotation.from_euler('xyz', angulos_euler, degrees=True)
    return rotacao.as_matrix()

if __name__ == "__main__":
    # Carregar a nuvem de pontos
    diretorio_atual = os.path.dirname(os.path.realpath(__file__))
    pcd = o3d.io.read_point_cloud(os.path.join(diretorio_atual, "../point_clouds/bunny2.ply"))
    frame_coordenado = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # Calcular o centro de gravidade da nuvem de pontos
    centro_gravidade = pcd.get_center()

    # Aplicar a translação para trazer o centro de gravidade para a origem
    pcd.translate(-centro_gravidade)
    
    # Salvar a nuvem de pontos original para comparação
    pcd_original = copy.deepcopy(pcd)
    
    # Visualizar a nuvem de pontos original
    o3d.visualization.draw_geometries([pcd_original, frame_coordenado], window_name="Nuvem de Pontos Original")

    # Definir ângulos de Euler
    angulos_euler = [45, 30, 60]  # Ângulos em graus para as sequências XYZ

    # Criar a matriz de rotação a partir dos ângulos de Euler
    matriz_rotacao = angulos_euler_para_matriz_rotacao(angulos_euler)

    # Criar a matriz homogênea
    matriz_homogenea = np.identity(4)
    matriz_homogenea[:3, :3] = matriz_rotacao
    
    # Adicionar translação
    translacao = np.array([1, 0, 0])
    matriz_homogenea[:3, 3] = translacao

    # Transformar a nuvem de pontos usando a matriz homogênea
    pcd.transform(matriz_homogenea)

    # Visualizar a nuvem de pontos transformada
    o3d.visualization.draw_geometries([pcd_original.paint_uniform_color([1.0, 0, 0]), 
                                       pcd.paint_uniform_color([0, 1.0, 0]),
                                       frame_coordenado], window_name="Nuvem de Pontos Transformada")
    