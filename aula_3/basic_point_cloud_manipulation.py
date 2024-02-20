import open3d as o3d
import os


# Função para criar e visualizar um voxel a partir de uma distância
def visualizar_voxel(pcd, voxel_size, window_name):
    pcd_voxel_filtrada = pcd.voxel_down_sample(voxel_size)
    o3d.visualization.draw_geometries([pcd_voxel_filtrada], window_name=window_name)
    
# Função para visualizar a bounding box alinhada aos eixos (AABB) e orientada (OBB)
def visualizar_bounding_boxes(pcd):
    # AABB (bounding box alinhada aos eixos)
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = [1, 0, 0]
    # OBB (bounding box orientada)
    obb = pcd.get_oriented_bounding_box()
    obb.color = [0, 1, 0]
    # Eixos coordenados
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([pcd, aabb, obb, coordinate_frame])
    
# Função para calcular o envoltório convexo e visualizar o resultado
def visualizar_convex_hull(pcd, window_name):
    convex_hull, _ = pcd.compute_convex_hull()
    convex_hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(convex_hull)
    convex_hull_ls.paint_uniform_color((1, 0, 0))
    o3d.visualization.draw_geometries([pcd, convex_hull_ls], window_name=window_name)
    
# Função para aplicar o filtro SOR e visualizar o resultado
def visualizar_filtro_sor(pcd, std_ratio, window_name):
    pcd_filtered, _ = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=std_ratio)
    o3d.visualization.draw_geometries([pcd, pcd_filtered.paint_uniform_color([0, 1.0, 0])], window_name=window_name)
    
def crop_nuvem_de_pontos(pcd):
    aabb = pcd.get_axis_aligned_bounding_box()
    # Calcular a metade da extensão no eixo x
    mid_x = (aabb.min_bound[0] + aabb.max_bound[0]) / 2
    # Definir a caixa delimitadora para o recorte
    crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=[mid_x, aabb.min_bound[1], aabb.min_bound[2]],
                                                    max_bound=aabb.max_bound)
    # Aplicar o recorte na nuvem de pontos original
    cropped_pcd = pcd.crop(crop_box)
    o3d.visualization.draw_geometries([pcd, cropped_pcd.paint_uniform_color([0, 1.0, 0])], window_name="Point Cloud Cropped")
    
def calcular_normais(pcd, window_name):
    pcd.estimate_normals()
    o3d.visualization.draw_geometries([pcd], window_name=window_name)

if __name__ == "__main__":
    # Carregar a nuvem de pontos
    diretorio_atual = os.path.dirname(os.path.realpath(__file__))
    pcd = o3d.io.read_point_cloud(os.path.join(diretorio_atual, "../point_clouds/bunny2.ply"))

    # Visualizar a nuvem de pontos original
    o3d.visualization.draw_geometries([pcd], window_name="Nuvem de Pontos Original")

    # Distância inicial mínima
    distancia_voxel_base = 0.001
    
    # Visualizar o voxel com tamanho duas vezes a distância média dos vizinhos
    visualizar_voxel(pcd, distancia_voxel_base * 2, "Voxel 2x")
    
    # Visualizar o voxel com tamanho quatro vezes a distância média dos vizinhos
    visualizar_voxel(pcd, distancia_voxel_base * 4, "Voxel 4x")
    
    # Visualizar o voxel com tamanho seis vezes a distância média dos vizinhos
    visualizar_voxel(pcd, distancia_voxel_base * 6, "Voxel 6x")
    
    # Visualizar as bounding boxes junto a point cloud
    visualizar_bounding_boxes(pcd)
    
    # Calcular o envoltório convexo e visualizar o resultado
    visualizar_convex_hull(pcd, "Envoltório Convexo")
    
    # Aplicar o filtro SOR na point cloud
    visualizar_filtro_sor(pcd, 0.5, "SOR")
    
    # Crop da nuvem de pontos
    crop_nuvem_de_pontos(pcd)
    
    # Calcular normais
    calcular_normais(pcd, "Nuvem com normais calculadas")
    