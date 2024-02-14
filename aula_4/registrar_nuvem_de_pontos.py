import open3d as o3d
import numpy as np
import os
import copy

def resultado_do_registro(fonte, alvo, transformacao):
    fonte_temp = copy.deepcopy(fonte)
    alvo_temp = copy.deepcopy(alvo)
    fonte_temp.paint_uniform_color([1, 0.706, 0])
    alvo_temp.paint_uniform_color([0, 0.651, 0.929])
    fonte_temp.transform(transformacao)
    o3d.visualization.draw_geometries([fonte_temp, alvo_temp], window_name="Nuvens de Pontos")

if __name__ == "__main__":
    # Carregar as nuvens de pontos
    diretorio_atual = os.path.dirname(os.path.realpath(__file__))
    pcd_cima = o3d.io.read_point_cloud(os.path.join(diretorio_atual, "../point_clouds/bunny2_up.ply"))
    pcd_baixo = o3d.io.read_point_cloud(os.path.join(diretorio_atual, "../point_clouds/bunny2_down.ply"))
    
    # Calcular normais das nuvens de pontos
    pcd_cima.estimate_normals()
    pcd_baixo.estimate_normals()
    
    # Alinhamento inicial
    transformacao_referencia = np.identity(4)
    # transformacao_referencia = np.array([[ 0.99842792,  0.05300482,  0.01822576, -0.13888624],
    #                                      [-0.04377151,  0.94043323, -0.33714891,  0.06385634],
    #                                      [-0.03501063,  0.33582112,  0.9412749,  -0.0289476 ],
    #                                      [ 0,           0,           0,           1         ]])
    resultado_do_registro(pcd_cima, pcd_baixo, transformacao_referencia)
    
    # Aplicar ICP para registro
    print("Aplicar ICP ponto a ponto")
    maxima_distancia_correspondencias = 0.01
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_cima, pcd_baixo, maxima_distancia_correspondencias, transformacao_referencia,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        # o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=300, relative_rmse=0.0001))
    print(reg_p2p)
    print("A transformação é:")
    print(reg_p2p.transformation)
    resultado_do_registro(pcd_cima, pcd_baixo, reg_p2p.transformation)
    