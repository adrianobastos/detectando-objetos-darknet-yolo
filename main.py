# Carrega as dependências
import cv2
import time

COLORS = [(0,255,255), (255,255,0), (0,255,0), (255,0,0)]

# Carrega as classes
class_names = []
with open("coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# captura do vídeo
cap = cv2.VideoCapture(0)

# Carregando os pesos da rede neural
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

# Setando os parâmetros da rede neural
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# Lendo os frames do vídeo
while True:

    # Captura do frame
    _, frame = cap.read()

    # Começo da contagem dos MS
    start = time.time()

    # Detecção
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)

    # Fim da contagem dos MS
    end = time.time()

    # Percorre todas as detecções
    for (classid, score, box) in zip(classes, scores, boxes):

        # Gerando um cor para a classe
        color = COLORS[int(classid) % len(COLORS)]

        # Pegando o nome da classe pelo ID e o seu Score de acuracia
        label = f"{class_names[classid]} : {score}"

        # Desenhando o box da detecção
        cv2.rectangle(frame, box, color, 2)

        # Escrevendo o nome da classe em cima do box do objeto
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Calculando o tempo que levou para fazer a detecção
    fps_label = f"FPS: {round((1.0/(end - start)),2)}"

    # Escrevendo o FPS na imagem
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Mostrando a imagem 
    cv2.imshow("Detections", frame)

    # Espera da resposta
    if cv2.waitKey(1) == 27:
        break

# Liberação da câmera e destrói todas as janelas
cap.release()
cv2.destroyAllWindows()