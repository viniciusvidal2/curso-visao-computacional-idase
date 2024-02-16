import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def detecta_keypoints_e_descritores(imagem):
    # Inicializa o detector SIFT
    sift = cv2.SIFT_create()

    # Encontra keypoints e descritores
    keypoints, descritores = sift.detectAndCompute(imagem, None)
    
    # Desenha os keypoints na imagem original
    imagem_com_keypoints = cv2.drawKeypoints(imagem, keypoints, None)

    # Cria uma imagem vazia para desenhar os descritores
    imagem_com_descritores = imagem.copy()

    # Desenha os descritores
    for kp in keypoints:
        x, y = kp.pt
        size = kp.size
        color = tuple([int(c) for c in np.random.randint(0, 255, 3)])
        cv2.circle(imagem_com_descritores, (int(x), int(y)), int(size), color, 2)

    return keypoints, descritores, imagem_com_keypoints, imagem_com_descritores

def match_e_filtragem_descritores(descritores_1, descritores_2, threshold=0.9):
    # Configurar o matcher de correspondência (usando força bruta)
    bf = cv2.BFMatcher()

    # Realizar correspondência entre os descritores das duas imagens
    correspondencias = bf.knnMatch(descritores_1, descritores_2, k=2)

    # Aplicar filtro de razão de Lowe para selecionar as melhores correspondências
    boas_correspondencias = []
    for m, n in correspondencias:
        if m.distance < threshold*n.distance:
            boas_correspondencias.append(m)

    return boas_correspondencias

def calcula_homografia(kpts_1, kpts_2, corresp):
    # Extrai os pontos correspondentes
    pontos1 = np.float32([kpts_1[m.queryIdx].pt for m in corresp]).reshape(-1, 1, 2)
    pontos2 = np.float32([kpts_2[m.trainIdx].pt for m in corresp]).reshape(-1, 1, 2)
    
    # Calcula a homografia usando RANSAC
    homografia, _ = cv2.findHomography(pontos1, pontos2, cv2.RANSAC)

    return homografia

def stitching_com_mascara(im1, im2, H):
    # Dimentsoes e corners das imagens
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    # Aplicar homografia nos corners para descobrir novo tamanho da imagem
    corners1_homografia = cv2.perspectiveTransform(corners1, H)
    # Novo tamanho das imagens pelo corners resultantes
    corners = np.concatenate((corners1_homografia, corners2), axis=0)
    [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)

    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    im1_homografia = cv2.warpPerspective(im1, Ht @ H, (xmax - xmin, ymax - ymin))
    im1_homografia[t[1]:h2 + t[1], t[0]:w2 + t[0]] = im2
    
    return im1_homografia.astype(np.uint8)
  
def main():
    # Obtém o diretório do arquivo atual
    diretorio_atual = os.path.dirname(os.path.realpath(__file__))
    print("Diretório do arquivo atual:", diretorio_atual)
    
    # Carrega as imagens e reduz resolucao
    imagem_1 = cv2.imread(os.path.join(diretorio_atual, '../datasets/stitching/drone_1.JPG'))
    imagem_2 = cv2.imread(os.path.join(diretorio_atual, '../datasets/stitching/drone_2.JPG'))
    h, w = imagem_1.shape[:2]
    nh = h // 5
    nw = w // 5
    imagem_1 = cv2.resize(imagem_1, (nw, nh))
    imagem_2 = cv2.resize(imagem_2, (nw, nh))
    
    # Mostra as imagens de entrada
    print("Analisando imagens iniciais ...")
    cv2.imshow('Imagem 1', imagem_1)
    cv2.imshow('Imagem 2', imagem_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Encontra keypoints e descritores
    print("Calculando keypoints e descritores ...")
    keypoints_1, descritores_1, imagem_1_com_keypoints, imagem_1_com_descritores = detecta_keypoints_e_descritores(imagem_1)
    keypoints_2, descritores_2, imagem_2_com_keypoints, imagem_2_com_descritores = detecta_keypoints_e_descritores(imagem_2)

    # Mostra as imagens com keypoints e descritores
    print("Mostrando imagens com keypoints ...")
    cv2.imshow('Imagem 1 com Keypoints', imagem_1_com_keypoints)
    cv2.imshow('Imagem 2 com Keypoints', imagem_2_com_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Mostrando imagens com descritores ...")
    cv2.imshow('Imagem 1 com Descritores', imagem_1_com_descritores)
    cv2.imshow('Imagem 2 com Descritores', imagem_2_com_descritores)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Realiza correspondência e filtragem dos descritores
    correspondencias = match_e_filtragem_descritores(descritores_1, descritores_2, 0.5)
    
    # Desenhar as correspondências
    resultado = cv2.drawMatches(imagem_1, keypoints_1, imagem_2, keypoints_2, correspondencias, 
                                None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Matches resultantes', resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Aplica a homografia na imagem 1
    print("Calcular e aplicar homografia da imagem 1 para a 2 ...")
    homografia = calcula_homografia(keypoints_1, keypoints_2, correspondencias)
    imagem_1_homografia = cv2.warpPerspective(imagem_1, homografia, (imagem_2.shape[1], imagem_2.shape[0]))
    cv2.imshow('Imagem 1 com homografia aplicada', imagem_1_homografia)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Realiza o stitching entre as imagens
    print("Aplicando stitching com mascara ...")
    resultado_stitching = stitching_com_mascara(imagem_1, imagem_2, homografia)
    cv2.imshow('Resultado do stitching', resultado_stitching)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main()
    