import numpy as np
import cv2
import math

# cap = cv2.VideoCapture('tenor.gif')
cap = cv2.VideoCapture('untitled.mp4')
# Define the codec and create VideoWriter object

fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 1.0, (640, 480))
first = True


# baseFrame = 670
# startingFrame = 4270 
# cap.set(2,startingFrame)
# cv2.namedWindow('Threshold',cv2.WINDOW_NORMAL)
# cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
# cv2.namedWindow('Gray',cv2.WINDOW_NORMAL)

# cv2.resizeWindow('Threshold', 600,600)
# cv2.resizeWindow('frame', 600,600)
# cv2.resizeWindow('Gray', 600,600)

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Image2', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 600, 600)
cv2.resizeWindow('Image2', 600, 600)

pontos = []
frameNum = 0

while cap.isOpened():
    ret, frame = cap.read()

    # Pula alguns frames pois demora para comecar
    frameNum = frameNum + 1
    if frameNum < 250:
        continue

    if first:
        frameCaminho = frame

    if ret:
        ptsNovos = np.array([], np.int32)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.equalizeHist( gray, gray );  
        circlesize = 21
        elipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (circlesize, circlesize))
        thresh = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, elipse)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # cv2.imshow("Gaussian Blur", gray)
        thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1] 
        thresh = cv2.erode(thresh, elipse, iterations=1)
        thresh = cv2.dilate(thresh, elipse, iterations=1)
        # thresh = cv2.bitwise_not(thresh) 
        # cv2.imshow("Gray", gray)
        # cv2.imshow("Threshold", thresh)
        img, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, cnts, -1, (0, 0, 255), 2)

        novosPontos = []

        for c in cnts:
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # draw the contour and center of the shape on the image
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
            cv2.circle(frame, (cX, cY), 7, (255, 0, 255), -1)
            cv2.putText(frame, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if first:
                novosPontos.append([cX, cY])
            else:
                novosPontos.append([cX, cY])
                menorDistancia = 99999999.
                pontoMaisPerto = []
                # Verifica qual o ponto anterior mais proximo do ponto atual
                for pt in pontos:
                    novaDistancia = math.sqrt(pow((cX-pt[0]), 2) + pow((cY-pt[1]), 2))
                    if novaDistancia < menorDistancia:
                        menorDistancia = novaDistancia
                        pontoMaisPerto = pt
                # Evita desenhar riscos absurdos
                if menorDistancia < 400:
                    cv2.line(frameCaminho, (cX, cY), (pontoMaisPerto[0], pontoMaisPerto[1]), (255, 255, 255), 5)

            # show the image
        cv2.imshow("Image", frame)
        cv2.imshow("Image2", frameCaminho)
        cv2.waitKey(1)

        # frame = cv2.flip(frame,0)

        # write the flipped frame
        # out.write(frame)

        # cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        pontos = novosPontos
        first = False
    else:
        break

# Release everything if job is finished
cap.release()
# out.release()
cv2.imwrite('resultado.png', frameCaminho)
cv2.destroyAllWindows()
