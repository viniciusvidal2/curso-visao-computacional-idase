import cv2
import numpy as np
import os
from time import sleep


def selecionar_pontos(imagem, im_id, num_pontos):
    pontos = []
    nome_janela = 'Imagem ' + str(im_id)

    def callback_do_mouse(event, x, y, flags, param):
        nonlocal pontos, nome_janela
        if event == cv2.EVENT_RBUTTONDOWN:
            pontos.append((x, y))
            cv2.circle(imagem, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(nome_janela, imagem)

    cv2.imshow(nome_janela, imagem)
    cv2.setMouseCallback(nome_janela, callback_do_mouse)

    while len(pontos) < num_pontos:
        cv2.waitKey(1)
    if len(pontos) == num_pontos:
        sleep(1)
            
    return np.asarray(pontos)

def stitching_com_mascara(im1, im2):
    # Criar máscara invertida para os pixels diferentes de zero na imagem1
    mascara = cv2.threshold(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY_INV)[1]

    # Aplicar a máscara invertida na imagem2
    im2_mascarada = cv2.bitwise_and(im2, im2, mask=mascara)
    
    cv2.imshow('Imagem 2 com Descritores', im2_mascarada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    # Somar a imagem1 com a imagem2 mascarada
    resultado = cv2.add(im1, im2_mascarada)
    
    return resultado

def main():
    diretorio_atual = os.path.dirname(os.path.realpath(__file__))
    print("Diretório do arquivo atual:", diretorio_atual)
    
    imagem_1 = cv2.imread(os.path.join(diretorio_atual, '../datasets/homography/perigo.jpg'))
    imagem_2 = cv2.imread(os.path.join(diretorio_atual, '../datasets/homography/transformador.jpg'))
    
    print("Analisando imagens iniciais ...")
    cv2.imshow('Imagem 1', imagem_1)
    cv2.imshow('Imagem 2', imagem_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Selectionar os pontos de match das imagens
    print("Selecione 4 pontos na imagem 1:")
    pontos1 = selecionar_pontos(imagem_1.copy(), 1, 4)
    print("Selecione 4 pontos na imagem 2:")
    pontos2 = selecionar_pontos(imagem_2.copy(), 2, 4)
    cv2.destroyAllWindows()
    
    # Calcular a homografia com base nos pontos
    print("Calculando homografia ...")
    homografia, _ = cv2.findHomography(pontos1, pontos2)
    imagem_1_homografia = cv2.warpPerspective(imagem_1, homografia, (imagem_2.shape[1], imagem_2.shape[0]))
    cv2.imshow('Imagem 1 com homografia aplicada', imagem_1_homografia)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Obter a transformacao em perspectiva
    print("Transformando imagens em perspectiva ...") 
    h, w = imagem_1.shape[:2]
    corners_1 = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    corners_1_homografia = cv2.perspectiveTransform(corners_1, homografia)
    imagem_2_copia = imagem_2.copy()
    cv2.polylines(imagem_2_copia, [np.int32(corners_1_homografia)], True, (0, 255, 0), 2)    
    cv2.imshow('Perspectiva da imagem 1 transformada na imagem 2', imagem_2_copia)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Realiza o stitching mesclando as imagens
    print("Aplicando stitching")
    resultado_stitching = stitching_com_mascara(imagem_1_homografia, imagem_2_copia)
    cv2.imshow('Resultado do stitching', resultado_stitching)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    