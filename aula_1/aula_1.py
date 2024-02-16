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

def main():
    # Obtém o diretório do arquivo atual
    diretorio_atual = os.path.dirname(os.path.realpath(__file__))
    print("Diretório do arquivo atual:", diretorio_atual)
    
    # Carrega as imagens
    imagem_1 = cv2.imread(os.path.join(diretorio_atual, '../datasets/fountain/0000.png'))
    imagem_2 = cv2.imread(os.path.join(diretorio_atual, '../datasets/fountain/0001.png'))
    
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


if __name__ == '__main__':
    main()
    