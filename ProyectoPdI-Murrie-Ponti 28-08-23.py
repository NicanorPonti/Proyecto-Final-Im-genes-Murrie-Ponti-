# ----------------------------------------------------------------------------------------------------------------------
# PROYECTO FINAL - INTRODUCCION AL PROCESAMIENTO DIGITAL DE IMAGENES
# ALEX MURRIE - PONTI GONZALEZ NICANOR
# AGOSTO 2023
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------- LIBRERIAS ------------------------------------------------------------------
import ctypes
import tkinter as tk
from tkinter import *
from tkinter import filedialog, messagebox
import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.signal import convolve2d
import math
from PIL import Image, ImageTk, ImageDraw
# ----------------------------------------------------------------------------------------------------------------------
M_YIQ = np.array([[0.299, 0.587, 0.114],  # Se cargan las matrices de conversion
                  [0.595716, -0.274453, -0.321263],  # RGB a YIQ
                  [0.211456, -0.522591, 0.311135]])

M_RGB = np.array([[1, 0.9563,  0.6210],               # YIQ a RGB
                  [1, -0.2721, -0.6474],
                  [1, -1.1070, 1.7046]])

def apply_matrix(img, M):                              # Función que acomoda el formato
    return np.matmul(img.reshape((-1, 3)), M.T).reshape(img.shape)

def rgb2yiq(img):                                      # Función que aplica la conversion rga a yiq
    return apply_matrix(img, M_YIQ)

def yiq2rgb(img):                                      # Función que aplica la conversion yiq a rgb
    return apply_matrix(img, M_RGB)

def plot_hist(im, bins, ax, cumulative=False):         # Función que grafica el histograma de una imagen
    counts, borders = np.histogram(im if im.ndim == 2 else rgb2yiq(im)[..., 0], bins=bins, range=(0, 1))
    ax.bar(range(len(counts)), np.cumsum(counts) if cumulative else counts)
    plt.xticks(ax.get_xticks(), labels=np.round(ax.get_xticks()/bins, 2))
    plt.grid(alpha=0.3)

def gaussian(N, sigma=1):
    x = np.linspace(-sigma, sigma, N+1) #linspace crea un vector de valores entre -sigma y sigma igualmente distribuidos
    gaussian_dist = np.diff(st.norm.cdf(x)) #CDF = Cumulative distribution function NORM: distribucion normal/gaussiana.
    gaussian_filter = np.outer(gaussian_dist, gaussian_dist)
    return gaussian_filter/gaussian_filter.sum()
# ----------------------------------------------------------------------------------------------------------------------
# Banderas para controlar el guardado de la imagen
flag_cargar = False
flag_filtrada = False
flag_bordes = False
flag_histograma = False
flag_suave = False
flag_seg = False #
flag_recorte = False
original = imageio
filename = ""  # Asigna un valor inicial vacío a filename
name = ""
# ----------------------------------------------------------------------------------------------------------------------
# Variables para Snake - Recortar - Etiquetar
canvas_vent_seg = FigureCanvasTkAgg
canvas_recortada = FigureCanvasTkAgg
xo = None
yo = None
ro = None
x = None
y = None
r = None
bw = None
lineas = []
circ_procesado = []
lineas_recorrido = []
points = None
circulo_snake = None
# ------------------------------------------------- FUNCIONES ----------------------------------------------------------
def load():
    global filename, canvas_vent_prin, lbl_orig, original, foto, image_tk, flag_cargar, image_x, image_y

    filename = filedialog.askopenfilename(initialdir="/Documentos",
                                          title="Seleccione una imagen",
                                          filetypes=(("png, jpg files", "*.png* *.jpg*"),
                                                     ("all files", "*.*")))
    if len(filename) > 0:
        foto = Image.open(filename)
        original = np.array(foto) / 255  # Convertir la imagen a una matriz NumPy
        flag_cargar = True
        # Redimensionar la imagen
        foto.thumbnail((450, 250))
        # Obtener las dimensiones del canvas
        canvas_width = 450
        canvas_height = 250
        # Calcular las coordenadas de posición para centrar la imagen
        image_x = (canvas_width - foto.width) // 2
        image_y = (canvas_height - foto.height) // 2
        image_tk = ImageTk.PhotoImage(foto)

        canvas_vent_prin= tk.Canvas(root, width=canvas_width, height=canvas_height, highlightthickness=0)
        canvas_vent_prin.place(x=250, y=52)
        canvas_vent_prin.configure(background=root.cget("background"))
        image_item = canvas_vent_prin.create_image(image_x, image_y, anchor=tk.NW, image=image_tk)

        lbl_orig = tk.Label(root, text="Original", background="white")
        lbl_orig.place(x=440, y=25)
        btn_load.config(state=tk.DISABLED)
        root.update()
    else:
        filename = "-"  # Es necesario para salvar excepciones de usuario

# ----------------------------------------------------------------------------------------------------------------------
def mostrar_histograma():
    global hist, flag_histograma, canvas_hist
    if len(filename) > 0:
        foto = Image.open(filename)
        original = np.array(foto)[:, :, 0:3] / 255
        lum = original.mean(axis=2)
        counts, bins = np.histogram(lum, 16)

        # Crear una nueva ventana para mostrar el histograma
        ventana_histograma = tk.Toplevel(root)
        ventana_histograma.title("Histograma")

        # Crear la figura y el lienzo para el histograma
        hist = plt.Figure()
        flag_histograma = True
        canvas_hist = FigureCanvasTkAgg(hist, master=ventana_histograma)
        canvas_hist.get_tk_widget().pack()

        # Graficar el histograma en la figura
        global ax
        ax = hist.add_subplot(111)
        ax.clear()
        ax.bar(bins[:-1], counts, width=np.diff(bins), align='edge')
        ax.set_xlabel('Valor de píxel')
        ax.xaxis.get_label().set_size(8)
        ax.set_ylabel('Frecuencia')
        ax.yaxis.get_label().set_size(8)
        ax.set_title('Histograma')
        canvas_hist.draw()  # Actualizar el lienzo
    else:
        messagebox.showinfo("Importante", "Primero debe insertar una imagen")

# ----------------------------------------------------------------------------------------------------------------------
def filtrado():
    global canvas_filtrada, lbl_filt, imgfilt, flag_filtrada
    if len(filename) > 1:
        btn_filt.config(state=tk.DISABLED)
        try:
            original = imageio.v2.imread(filename)[:, :, 1] / 255
            kernel = gaussian(15)
            imgfilt = convolve2d(original, kernel, mode='full')
            imgfilt = (imgfilt * 255).astype(np.uint8)# Ajustar el rango de valores de imgfilt a [0, 255]
            # Calcular las dimensiones del canvas de la imagen original
            canvas_width = canvas_vent_prin.winfo_width()
            canvas_height = canvas_vent_prin.winfo_height()

            # Calcular la relación de aspecto de la imagen
            aspect_ratio = imgfilt.shape[1] / imgfilt.shape[0]

            # Calcular las dimensiones de la imagen filtrada manteniendo la relación de aspecto
            if aspect_ratio > 1:
                imgfilt_width = canvas_width
                imgfilt_height = int(canvas_width / aspect_ratio)
            else:
                imgfilt_width = int(canvas_height * aspect_ratio)
                imgfilt_height = canvas_height

            # Redimensionar la imagen filtrada utilizando LANCZOS
            imgfilt_resized = Image.fromarray(imgfilt).resize((imgfilt_width, imgfilt_height), Image.LANCZOS)

            fig1 = Figure()
            a1 = fig1.subplots(1, 1)
            a1.axis('off')
            a1.imshow(imgfilt_resized, 'gray')
            fig1.subplots_adjust(left=0, right=1, bottom=0, top=1)
            canvas_filtrada = FigureCanvasTkAgg(fig1, master=root)
            canvas_filtrada.get_tk_widget().place(x=785, y=52, width=canvas_width, height=canvas_height)
            flag_filtrada = True

            lbl_filt = tk.Label(root, text="Filtrada", background="white")
            lbl_filt.place(x=975, y=25)
            canvas_filtrada.draw()
            root.update()
        except IndexError:
            messagebox.showinfo("Importante",
                                "La imagen que esta tratando de filtrar ya ha sido procesada o no cumple con los "
                                "requisitos necesarios para ser procesada (ej: 3 dimensiones).")
    else:
        messagebox.showinfo("Importante", "Primero debe insertar una imagen.")

# ----------------------------------------------------------------------------------------------------------------------
def suavizado():
    global canvas_suavizado, lbl_suave, img_suave_resized, flag_suave
    if len(filename) > 1:
        btn_suav.config(state=tk.DISABLED)
        try:
            original = imageio.v2.imread(filename)[:, :, 1] / 255
            kernel_enfoque = np.array([[0, -1, 0], [-1, 10, -1], [0, -1, 0]])
            img_suave = convolve2d(original, kernel_enfoque, mode='same', boundary='symm')
            # Ajustar el rango de valores de la imagen suavizada para mejorar la visualización
            img_suave_adjusted = (img_suave - np.min(img_suave)) / (np.max(img_suave) - np.min(img_suave))

            # Calcular las dimensiones del canvas de la imagen
            canvas_width = canvas_vent_prin.winfo_width()
            canvas_height = canvas_vent_prin.winfo_height()

            # Calcular la relación de aspecto de la imagen
            aspect_ratio = img_suave_adjusted.shape[1] / img_suave_adjusted.shape[0]

            # Calcular las dimensiones de la imagen manteniendo la relación de aspecto
            if aspect_ratio > 1:
                img_suave_width = canvas_width
                img_suave_height = int(canvas_width / aspect_ratio)
            else:
                img_suave_width = int(canvas_height * aspect_ratio)
                img_suave_height = canvas_height
            # Redimensionar la imagen utilizando LANCZOS
            img_suave_resized = Image.fromarray((img_suave_adjusted * 255).astype(np.uint8)).resize(
                (img_suave_width, img_suave_height), Image.LANCZOS)

            fig3 = Figure()
            a3 = fig3.subplots(1, 1)
            a3.axis('off')
            a3.imshow(img_suave_resized, 'gray')
            fig3.subplots_adjust(left=0, right=1, bottom=0, top=1)
            canvas_suavizado = FigureCanvasTkAgg(fig3, master=root)
            canvas_suavizado.get_tk_widget().place(x=250, y=365, width=canvas_width, height=canvas_height)
            flag_suave = True
            lbl_suave = tk.Label(root, text="Suavizada", background="white")
            lbl_suave.place(x=440, y=340)
            canvas_suavizado.draw()
            root.update()
        except IndexError:
            messagebox.showinfo("Importante","La imagen que esta tratando de suavizar ya ha sido procesado o no cumple "
                                             "con los requisitos necesarios para ser procesada (ej: 3 dimensiones).")
    else:
        messagebox.showinfo("Importante", "Primero debe insertar una imagen.")

# ----------------------------------------------------------------------------------------------------------------------
def det_sobel():# ---------- detector de bordes ----------
    global canvas_bordes, lbl_bordes, detector, original, flag_bordes
    kernelX = np.array([[-1, 0, 1], [-2, 0, 2],[-1, 0, 1]])# Gradiente de x
    kernelY = np.array([[1, 2, 1], [0, 0, 0],[-1, -2, -1]])# Gradiente de y
    if len(filename) > 1:
        btn_bordes.config(state=tk.DISABLED)
        try:
            original = imageio.v2.imread(filename)[:, :, 1] / 255
            flag_bordes = True
            img_filt_X = convolve2d(original, kernelX, mode='same', boundary='symm')  # convolución por gradiente
            img_filt_Y = convolve2d(original, kernelY, mode='same', boundary='symm')
            modulo = np.sqrt(img_filt_X * img_filt_X + img_filt_Y * img_filt_Y) # Combinación de ambos gradientes
            umbral = 0.5
            detector = modulo
            for i in range(len(modulo)):  # Recorro la matriz y comparo con el umbral
                for j in range(len(modulo[0])):
                    if (modulo[i][j] < umbral):
                        detector[i][j] = 0  # Si el módulo de la imagen no supera el umbral le pongo 0
                    else:
                        detector[i][j] = 1  # Si el módulo de la imagen supera el umbral queda en 1

            # Ajustar el rango de valores a [0, 255]
            detector = (detector * 255).astype(np.uint8)
            # Calcular las dimensiones del canvas de la imagen
            canvas_width = canvas_vent_prin.winfo_width()
            canvas_height = canvas_vent_prin.winfo_height()

            # Calcular la relación de aspecto de la imagen
            aspect_ratio = detector.shape[1] / detector.shape[0]

            # Calcular las dimensiones de la imagen manteniendo la relación de aspecto
            if aspect_ratio > 1:
                detector_width = canvas_width
                detector_height = int(canvas_width / aspect_ratio)
            else:
                detector_width = int(canvas_height * aspect_ratio)
                detector_height = canvas_height

            # Redimensionar la imagen utilizando LANCZOS
            detector_resized = Image.fromarray(detector).resize((detector_width, detector_height), Image.LANCZOS)

            fig2 = Figure()
            a2 = fig2.subplots(1, 1)
            a2.axis('off')
            a2.imshow(detector_resized, 'gray')
            fig2.subplots_adjust(left=0, right=1, bottom=0, top=1)
            canvas_bordes = FigureCanvasTkAgg(fig2, master=root)
            canvas_bordes.get_tk_widget().place(x=785, y=365, width=canvas_width, height=canvas_height)

            lbl_bordes = tk.Label(root, text="Detector de bordes sobre la original", background="white")
            lbl_bordes.place(x=910, y=340)
            canvas_bordes.draw()
            root.update()
        except IndexError:
            messagebox.showinfo("Importante",
                                "La imagen a la que esta tratando de detectar sus bordes ya ha sido procesada "
                                "o no cumple con los requisitos necesarios para ser procesada (ej: 3 dimensiones).")
    else:
        messagebox.showinfo("Importante", "Primero debe insertar una imagen.")

# --------------- Funciones para guardar las distintas imagenes procesadas -----------------
def guardar_orig():
    global foto
    if flag_cargar:
        root.f = filedialog.asksaveasfilename(title="Guardar como",
                                              initialdir='/Escritorio',
                                              defaultextension=".png",
                                              filetypes=(("png files", "*.png*"), ("all files", "*.*")))
        if root.f:# Verificar si se proporcionó un nombre válido para que no se guarde si no se especificó nombre
            imageio.imwrite(root.f, foto)
    else:
        messagebox.showinfo("Importante", "Primero debe insertar una imagen")

def guardar_hist():
    global canvas_hist
    if len(filename) > 1:
        if flag_histograma:
            root.f = filedialog.asksaveasfilename(title="Guardar como",
                                                  initialdir='/Escritorio',
                                                  defaultextension=".png",
                                                  filetypes=(("png files", "*.png*"), ("all files", "*.*")))
            if root.f:  # Verificar si se proporcionó un nombre válido para que no se guarde si no se especificó nombre
                figura = canvas_hist.figure
                figura.savefig(root.f)
        else:
            messagebox.showinfo("Importante", "Debe abrir el histograma.\nOpciones --> Ver Histograma")
    else:
        messagebox.showinfo("Importante", "Primero debe insertar una imagen y abrir el histograma")

def guardar_filt():
    global imgfilt
    if flag_filtrada:
        root.f = filedialog.asksaveasfilename(title="Guardar como",
                                              initialdir='/Escritorio',
                                              defaultextension=".png",
                                              filetypes=(("png files", "*.png*"), ("all files", "*.*")))
        if root.f:# Verificar si se proporcionó un nombre válido para que no se guarde si no se especificó nombre
            imageio.imwrite(root.f, imgfilt)
    else:
        messagebox.showinfo("Importante", "Primero debe filtrar la imagen")

def guardar_suave():
    global img_suave_resized
    if flag_suave:
        root.f = filedialog.asksaveasfilename(title="Guardar como",
                                              initialdir='/Escritorio',
                                              defaultextension=".png",
                                              filetypes=(("png files", "*.png*"), ("all files", "*.*")))
        if root.f:# Verificar si se proporcionó un nombre válido para que no se guarde si no se especificó nombre
            imageio.imwrite(root.f, img_suave_resized)
    else:
        messagebox.showinfo("Importante", "Primero debe suavizar la imagen")

def guardar_borde():
    global detector
    if flag_bordes:
        root.f = filedialog.asksaveasfilename(title="Guardar como",
                                              initialdir='/Escritorio',
                                              defaultextension=".png",
                                              filetypes=(("png files", "*.png*"), ("all files", "*.*")))
        if root.f:# Verificar si se proporcionó un nombre válido para que no se guarde si no se especificó nombre
            imageio.imwrite(root.f, detector)
    else:
        messagebox.showinfo("Importante", "Primero debe detectar los bordes de la imagen")

# Función para limpiar la ventana
def destroy():
    global flag_cargar, flag_filtrada, flag_bordes, flag_histograma, filename, flag_suave
    if filename != "":
        if flag_cargar:
            canvas_vent_prin.delete("all")
            lbl_orig.destroy()
            flag_cargar = False
            flag_histograma = False
        if flag_filtrada:
            canvas_filtrada.get_tk_widget().destroy()
            lbl_filt.destroy()
            canvas_filtrada.draw()
            flag_filtrada = False
            btn_filt.config(state=tk.NORMAL)
        if flag_bordes:
            canvas_bordes.get_tk_widget().destroy()
            lbl_bordes.destroy()
            canvas_bordes.draw()
            flag_bordes = False
            btn_bordes.config(state=tk.NORMAL)
        if flag_suave:
            canvas_suavizado.get_tk_widget().destroy()
            lbl_suave.destroy()
            canvas_suavizado.draw()
            flag_suave = False
            btn_suav.config(state=tk.NORMAL)
        btn_load.config(state=tk.NORMAL)
        btn_filt.config(state=tk.NORMAL)
        btn_suav.config(state=tk.NORMAL)
        btn_bordes.config(state=tk.NORMAL)
        filename = ""
        root.update()
    else:
        messagebox.showinfo("Importante", "La ventana ya esta limpia.")

# Función para el atajo que se activa cuando se presiona ESC en el teclado
def exit(event):
    root.destroy()

# ----------------------- Funciones para abrir una nueva ventana que aplica el segmentador -----------------------------
# ---------------------------------- Funciones para aplicar segmentación con Snake -------------------------------------
def abrir_ventana_segmentacion():

    # --------------- Funcion que carga la imagen a segmentar -----------------

    def cargar_para_seg():
        global name, canvas_vent_seg, canvas_recortada, lbl1, bw, original1

        name = filedialog.askopenfilename(initialdir="/Documentos",
                                              title="Seleccione una imagen",
                                              filetypes=(("png, jpg files", "*.png* *.jpg*"),
                                                         ("all files", "*.*")))
        if len(name) > 0:
            try:
                original1 = imageio.v2.imread(name)[:, :, 0:3] / 255
                bw = rgb2yiq(original1)[:, :, 0]
                bw = np.clip(bw, 0, 1)
                fig = Figure()
                a = fig.add_axes([0, 0, 1, 1], frameon=False)
                a.axis('off')
                a.imshow(original1, cmap='gray', interpolation='nearest')
                canvas_vent_seg = FigureCanvasTkAgg(fig, master=ventana_segmentacion)
                canvas_vent_seg.get_tk_widget().place(x=250, y=60, width=bw.shape[1], height=bw.shape[0])
                canvas_vent_seg.draw()

            except IndexError:
                original1 = imageio.v2.imread(name) / 255
                original_colorized = np.zeros((original1.shape[0], original1.shape[1], 3), dtype=np.uint8)
                # Se rellena la matriz con gris para conservar las 3 dimensiones
                original_colorized[:, :, 0] = original1 * 128
                original_colorized[:, :, 1] = original1 * 128
                original_colorized[:, :, 2] = original1 * 128

                bw = rgb2yiq(original_colorized)[:, :, 0]
                bw = np.clip(bw, 0, 1)
                fig = Figure()
                a = fig.add_axes([0, 0, 1, 1], frameon=False)
                a.axis('off')
                a.imshow(original_colorized, cmap='gray', interpolation='nearest')
                canvas_vent_seg = FigureCanvasTkAgg(fig, master=ventana_segmentacion)
                canvas_vent_seg.get_tk_widget().place(x=250, y=60, width=bw.shape[1], height=bw.shape[0])
                canvas_vent_seg.draw()

            lbl1 = tk.Label(ventana_segmentacion, text="Imagen original", background="white")
            lbl1.place(x=200+bw.shape[1]/2, y=25)
            btnseg_cargar.config(state=tk.DISABLED)
            btn_proc.config(state=tk.DISABLED)
            btn_seg.config(state=tk.DISABLED)
            btn_recorte.config(state=tk.DISABLED)
            btn_borr.config(state=tk.DISABLED)
            btn_eti.config(state=tk.DISABLED)
            ventana_segmentacion.update()
        else:
            name = "-"  # Es necesario para salvar excepciones de usuario

    # --------------- Funcion que habilita el onclick de mouse e iniciar el ciculo -----------------
    def dib_circ():
        if len(name) > 1:
            ventana_segmentacion.bind('<Button-1>', getorigen)
            ventana_segmentacion.update()
        else:
            messagebox.showinfo("Importante", "Primero debe insertar una imagen.", parent=ventana_segmentacion)

    # --------------- Funcion que obtiene el origen del circulo -----------------
    def getorigen(event):
        global xo, yo, circulo_snake
        xo, yo = 0, 0
        xo, yo = event.x, event.y
        ventana_segmentacion.bind('<Motion>', getxy)                #Se habilitan la funcion de mover
        ventana_segmentacion.bind('<ButtonRelease-1>', final_circ)  # y soltar, las cuales corresponden
        ventana_segmentacion.update()                               # al radio y seteo del circulo

    # --------------- Funcion que obtiene las coordenadas de mouse y redibuja el circulo-----------------
    def getxy(event):
        global xo, yo, x, y, circulo_snake, ro
        if circulo_snake is not None:
            try:
                canvas_vent_seg.get_tk_widget().delete(circulo_snake) #En caso de ser la primera vez, no lo borra porque no existe
            except:
                print("Error")

        if xo is not None and yo is not None:                         #Caso contrario lo dibuja en las coordenadas iniciales con el radio del mouse
            x, y = event.x, event.y
            ro = (((x - xo) ** 2) + ((y - yo) ** 2)) ** 0.5
            circulo_snake = canvas_vent_seg.get_tk_widget().create_oval((xo - ro), (yo - ro), (xo + ro), (yo + ro), width=3, outline='blue')
            canvas_vent_seg.draw()
            if bw[y, x] < 0.1:
                canvas_vent_seg.get_tk_widget().create_oval(x, y, x, y, width=3, outline='black')

    # --------------- Funcion que setea el circulo final y deshabilita las funciones del mouse para evitar inconvenientes -----------------
    def final_circ(event):
        global xo, yo, ro, circulo_snake
        ventana_segmentacion.unbind('<Button-1>')
        ventana_segmentacion.unbind('<Motion>')
        ventana_segmentacion.unbind('<ButtonRelease-1>')
        if xo is not None and yo is not None:                           #Aqui se borra y redibuja el circulo por ultima vez
            canvas_vent_seg.get_tk_widget().delete(circulo_snake)
            x, y = event.x, event.y
            ro = int((((x - xo) ** 2) + ((y - yo) ** 2)) ** 0.5)
            circulo_snake = canvas_vent_seg.get_tk_widget().create_oval((xo - ro), (yo - ro), (xo + ro), (yo + ro), width=3,outline='blue')
            canvas_vent_seg.draw()
        btn_proc.config(state=tk.NORMAL)
        btn_borr.config(state=tk.NORMAL)
        btn_circ.config(state=tk.DISABLED)

    # La función procesar toma el centro y el radio de un círculo previamente definido y calcula las coordenadas de puntos en
    # la circunferencia utilizando trigonometría. Luego, dibuja estos puntos en el lienzo
    def procesar():
        global xo, yo, ro, points, circ_procesado, circ_reference
        points = []
        if len(name) > 1:
            for i in range(alpha.get()):                        #Aqui se crean los puntos limite que analizan la figura
                angle = i * (2 * math.pi / alpha.get())         # en funcion de la cantidad ingresada por el usuario
                x = int(xo + ro * math.cos(angle))              # Los puntos son equidistantes como se aprecia en el
                y = int(yo + ro * math.sin(angle))              # algoritmo, se generan cada 360/n (n = alpha.get())
                points.append([x, y, 0])
                circ_reference = canvas_vent_seg.get_tk_widget().create_oval(x - 1, y - 1, x + 1, y + 1, width=4, outline='red')
                circ_procesado.append(circ_reference)
                btn_seg.config(state=tk.NORMAL)
                btn_proc.config(state=tk.DISABLED)
                alpha_slider.config(state=tk.DISABLED)
        else:
            messagebox.showinfo("Importante", "Primero debe insertar una imagen.", parent=ventana_segmentacion)

    # --------------- Funcion encargada de la segmentacion como tal.       -----------------
    # --------------- En ella se reduce el radio de los puntos generados   -----------------
    # --------------- previamente al mismo tiempo que se analiza el limite -----------------
    # --------------- entre el pixel previo y el actual.                   -----------------
    def segmentar():
        global xo, yo, ro, points, lineas, flag_seg, lineas_recorrido
        rmax = 0
        if len(name) > 1:
            r = ro
            new_points = points.copy()
            auxiliar = points.copy()
            control = 0
            j = 0
            while (control == 0) and (r != 1):                      #Bucle de control que verifica puntos finalizados o radio distinto de 1
                j = j + 1
                control = 1
                r = r - 1
                for i in range(alpha.get()):                        #Bucle que analiza punto por punto ya generado y actualiza su posicion
                    if new_points[i][2] == 0:
                        angle = i * (2 * math.pi / alpha.get())
                        x = int(xo + r * math.cos(angle))
                        y = int(yo + r * math.sin(angle))
                        try:                                        #Aqui se realiza el control entre pixeles basado en el limite ingreado por el usuario
                            if (bw[new_points[i][1]][new_points[i][0]] + gamma.get()) > bw[y][x] > (bw[new_points[i][1]][new_points[i][0]] - gamma.get()):
                                new_points[i] = [x, y, 0]
                                auxiliar[i] = [x, y, 0]
                                linea_roja = canvas_vent_seg.get_tk_widget().create_oval(x - 1, y - 1, x + 1, y + 1, width=2, outline='red')
                            else:
                                if rmax == 0:
                                    rmax = r
                                new_points[i] = [x, y, 1]
                                linea_roja = canvas_vent_seg.get_tk_widget().create_oval(x - 1, y - 1, x + 1, y + 1, width=2, outline='red')
                            dist = np.sqrt((xo - x) ** 2 + (yo - y) ** 2)

                            if rmax != 0:
                                if rmax*(1+beta.get()) > dist > rmax*(1-beta.get()):
                                    pass
                                else:
                                    new_points[i] = [x, y, 1]
                                    linea_roja = canvas_vent_seg.get_tk_widget().create_oval(x - 1, y - 1, x + 1, y + 1, width=2, outline='red')

                            lineas_recorrido.append(linea_roja)
                            control = control * new_points[i][2]
                        except IndexError:
                            messagebox.showinfo("Error", "El circulo excede los tamaños permitidos para poder "
                                                         "segmentarlo.\nEl total del circulo debe estar contenido "
                                                         "en la imagen.", parent=ventana_segmentacion)
                            return
            lineas = []
            for i in range(alpha.get()):
                j = i + 1
                if i == alpha.get() - 1:
                    j = 0
                linea_id = canvas_vent_seg.get_tk_widget().create_line(new_points[i][0], new_points[i][1], new_points[j][0],
                                                             new_points[j][1], fill="red", width=2)
                lineas.append(linea_id)
            flag_seg = True
            btn_seg.config(state=tk.DISABLED)
            btn_recorte.config(state=tk.NORMAL)
            beta_slider.config(state=tk.DISABLED)
            gamma_slider.config(state=tk.DISABLED)
        else:
            messagebox.showinfo("Importante", "Primero debe insertar una imagen.", parent=ventana_segmentacion)

    # --------------- Funcion que recorta los pixeles encerrados por la segmentacion final -----------------
    def recortar():
        global xo, yo, ro, canvas_recortada, lineas, flag_recorte, recorte
        puntos_recorte = []
        coords_x = []
        coords_y = []
        dist_x = []
        dist_y = []

        if len(name) > 1:
            if len(original1.shape) == 3 and original1.shape[2] == 3:   # Imagen a color
                recorte = original1.copy()
                for i in range(len(recorte[:, 0, 0])):
                    for j in range(len(recorte[0, :, 0])):
                        recorte[i, j] = [1, 1, 1]
            else:                                                       # Imagen en blanco y negro
                recorte = original1.copy()
                recorte[:, :] = 1                                       # Rellenar con valor blanco

            for i in range(alpha.get()):
                coordenadas = canvas_vent_seg.get_tk_widget().coords(lineas[i])
                coords_x.append(int(coordenadas[0]))
                coords_x.append(int(coordenadas[2]))
                coords_y.append(int(coordenadas[1]))
                coords_y.append(int(coordenadas[3]))
                dist_x.append(int(coordenadas[0] - coordenadas[2]))
                dist_y.append(int(coordenadas[1] - coordenadas[3]))

            for i in range(alpha.get()):        #Algoritmo de Bresenham para estimar los pixeles de la rectas que encierran los puntos
                x0 = coords_x[2 * i]
                x1 = coords_x[2 * i + 1]
                y0 = coords_y[2 * i]
                y1 = coords_y[2 * i + 1]

                dx = abs(x1 - x0)
                dy = abs(y1 - y0)
                sx = -1 if x0 > x1 else 1
                sy = -1 if y0 > y1 else 1
                err = dx - dy

                while True:                     # Se redibujan las lineas con el algorimo de Bresenham
                    puntos_recorte.append([y0, x0])
                    canvas_vent_seg.get_tk_widget().create_oval(x0, y0, x0 + 1, y0 + 1, outline='blue', width=2)
                    if x0 == x1 and y0 == y1:
                        break
                    e2 = 2 * err
                    if e2 > -dy:
                        err -= dy
                        x0 += sx
                    if e2 < dx:
                        err += dx
                        y0 += sy


            puntos_recorte = np.array(puntos_recorte)
            maximo_vert = np.max(puntos_recorte[:, 0])
            minimo_vert = np.min(puntos_recorte[:, 0])
            iter_vert = maximo_vert - minimo_vert
            max_hor = 0
            min_hor = 10000

            for i in range(iter_vert):                  #Bucle de control que analiza todos los puntos dentro de los valores maximos del recorte
                indice = np.where(puntos_recorte[:, 0] == i + minimo_vert)
                maximo_horizont = np.max(puntos_recorte[indice, 1])
                minimo_horizont = np.min(puntos_recorte[indice, 1])
                iter_horizont = maximo_horizont - minimo_horizont

                if maximo_horizont > max_hor:
                    max_hor = maximo_horizont
                if minimo_horizont < min_hor:
                    min_hor = minimo_horizont

                for j in range(iter_horizont):          # Solo se guardan los puntos encerrados por las lineas de la segmentacion
                    recorte[i + minimo_vert][j + minimo_horizont] = original1[i + minimo_vert][j + minimo_horizont]

            try:
                if minimo_vert > 20 and min_hor > 20:
                    recorte = recorte[minimo_vert - 20:maximo_vert + 20, min_hor - 20:max_hor + 20]
                else:
                    recorte = recorte[minimo_vert:maximo_vert, min_hor:max_hor]
            except:
                pass

            flag_recorte = True
            fig1 = Figure()
            a = fig1.add_axes([0, 0, 1, 1], frameon=False)
            a.axis('off')
            a.imshow(recorte, cmap='gray', interpolation='nearest')
            canvas_recortada = FigureCanvasTkAgg(fig1, master=ventana_segmentacion)
            canvas_recortada.get_tk_widget().place(x=bw.shape[1] + 350, y=60, width=recorte.shape[1], height=recorte.shape[0])
            canvas_recortada.draw()
            btn_recorte.config(state=tk.DISABLED)
            btn_eti.config(state=tk.NORMAL)
        else:
            messagebox.showinfo("Importante", "Primero debe insertar una imagen.", parent=ventana_segmentacion)

    # Función que guarda la imagen recortada
    def etiquetar():
        global recorte
        if len(name) > 1:
            if flag_recorte:
                root.f = filedialog.asksaveasfilename(title="Guardar como",
                                                      initialdir='/Escritorio',
                                                      defaultextension=".png",
                                                      filetypes=(("png files", "*.png*"), ("all files", "*.*")))
                if root.f:
                    etiquetada = Image.fromarray((recorte * 255).astype('uint8'))
                    imagen_procesada = etiquetada.copy()
                    imagen_procesada.save(root.f)
                    btn_eti.config(state=tk.DISABLED)
                else:
                    messagebox.showinfo("Importante", "Seleccione un nombre para la imagen.", parent=ventana_segmentacion)
            else:
                messagebox.showinfo("Importante", "Primero debe recortar la imagen.", parent=ventana_segmentacion)
        else:
            messagebox.showinfo("Importante", "Primero debe insertar una imagen.", parent=ventana_segmentacion)

    # Función para guardar la imagen original con los procesos que se hacen sobre ella
    def guardar_seg():
        global lineas, original1, circulo_snake, lineas_recorrido
        if flag_seg:
            root.f = filedialog.asksaveasfilename(title="Guardar como",
                                                  initialdir='/Escritorio',
                                                  defaultextension=".png",
                                                  filetypes=(("png files", "*.png*"), ("all files", "*.*")))
            if root.f:
                # Convierte la matriz NumPy en una imagen PIL para poder dibujar los procesos que se ven en la
                # interfaz y se guarde la imagen con ello
                original1_pil = Image.fromarray((original1 * 255).astype('uint8'))
                imagen_procesada = original1_pil.copy()
                draw = ImageDraw.Draw(imagen_procesada)

                for linea_id in lineas:
                    coords = canvas_vent_seg.get_tk_widget().coords(linea_id)
                    draw.line(coords, fill='red', width=2)
                circulo_coords = canvas_vent_seg.get_tk_widget().coords(circulo_snake)
                circulo_bbox = (circulo_coords[0], circulo_coords[1], circulo_coords[2], circulo_coords[3])
                ImageDraw.Draw(imagen_procesada).ellipse(circulo_bbox, outline='blue', width=3)
                for linea_roja in lineas_recorrido:
                    oval_coords = canvas_vent_seg.get_tk_widget().coords(linea_roja)
                    oval_bbox = (oval_coords[0], oval_coords[1], oval_coords[2], oval_coords[3])
                    ImageDraw.Draw(imagen_procesada).ellipse(oval_bbox, outline='red', width=2)
                imagen_procesada.save(root.f)
        else:
            messagebox.showinfo("Importante", "Primero debe insertar una imagen y segmentarla", parent=ventana_segmentacion)

    # Función para deshacer el círculo y/o las lineas que hacen el segmentado
    def borrar_seg():
        global circulo_snake, circ_reference, circ_procesado, lineas, lineas_recorrido, name, flag_seg, flag_recorte
        if name != "":
            if circulo_snake is not None:
                canvas_vent_seg.get_tk_widget().delete(circulo_snake)
                circulo_snake = None
                btn_circ.config(state=tk.NORMAL)
                btn_proc.config(state=tk.DISABLED)
            if circ_procesado != []:
                for circ_reference in circ_procesado:
                    canvas_vent_seg.get_tk_widget().delete(circ_reference)
                circ_procesado = []
                btn_circ.config(state=tk.NORMAL)
                btn_proc.config(state=tk.DISABLED)
                btn_seg.config(state=tk.DISABLED)
            if lineas != [] and lineas_recorrido != []: #si se apretó segmentar
                for linea_id in lineas:
                    canvas_vent_seg.get_tk_widget().delete(linea_id)
                lineas = []
                for linea_roja in lineas_recorrido:
                    canvas_vent_seg.get_tk_widget().delete(linea_roja)
                lineas_recorrido = []
                btn_recorte.config(state=tk.DISABLED)
            if flag_recorte:
                canvas_recortada.get_tk_widget().destroy()
                canvas_recortada.draw()
                btn_eti.config(state=tk.DISABLED)
            alpha_slider.config(state=tk.NORMAL)
            beta_slider.config(state=tk.NORMAL)
            gamma_slider.config(state=tk.NORMAL)
            flag_seg = False
            flag_recorte = False
            ventana_segmentacion.update()
        else:
            messagebox.showinfo("Importante", "La ventana ya esta limpia.", parent=ventana_segmentacion)

    # Función para limpiar la ventana
    def limpiar():
        global name, flag_seg, flag_recorte
        if name != "":
            canvas_vent_seg.get_tk_widget().destroy()
            canvas_vent_seg.draw()
            lbl1.destroy()
            if flag_recorte:
                canvas_recortada.get_tk_widget().destroy()
                canvas_recortada.draw()
            flag_seg = False
            flag_recorte = False
            btnseg_cargar.config(state=tk.NORMAL)
            btn_circ.config(state=tk.NORMAL)
            btn_proc.config(state=tk.NORMAL)
            btn_seg.config(state=tk.NORMAL)
            btn_recorte.config(state=tk.NORMAL)
            btn_eti.config(state=tk.NORMAL)
            alpha_slider.config(state=tk.NORMAL)
            beta_slider.config(state=tk.NORMAL)
            gamma_slider.config(state=tk.NORMAL)
            ventana_segmentacion.update()
            name = ""
        else:
            messagebox.showinfo("Importante", "La ventana ya esta limpia.", parent=ventana_segmentacion)

    def instrucciones():
        messagebox.showinfo("Instrucciones", "Para realizar un circulo debe hacer click al centro del futuro "
                                        "circulo y arrastrar el mouse. Luego puede personalizar los parámetros "
                                        "y en caso de equivocarse a lo largo del proceso "
                                        "se puede presionar el botón 'Deshacer' y volver a realizar el circulo "
                                             "(luego de presionar el botón 'Dibujar circulo')."
                                        "\n\nDebe limpiar la ventana si quiere cargar una nueva imagen.",
                            parent=ventana_segmentacion)

   #--------------------------------------- INTERFAZ DE VENTANA SEGMENTACION -------------------------------------------
    ventana_segmentacion = tk.Toplevel(root)
    ventana_segmentacion.title("Segmentador")
    ventana_segmentacion.geometry("1100x600")
    ventana_segmentacion.configure(bg='white')
    ventana_segmentacion.state('zoomed')

    # ----------- BOTONES -------------
    btnseg_cargar = tk.Button(ventana_segmentacion, text="Cargar imagen", background="lightblue", command=cargar_para_seg)
    btnseg_cargar.place(x=20, y=20)

    btn_circ = tk.Button(ventana_segmentacion, text="Dibujar circulo", background="lightblue", command=dib_circ)
    btn_circ.place(x=20, y=65)

    btn_proc = tk.Button(ventana_segmentacion, text="Procesar", background="lightblue", command=procesar)
    btn_proc.place(x=20, y=110)

    btn_borr = tk.Button(ventana_segmentacion, text="Deshacer", background="yellowgreen", command=borrar_seg)
    btn_borr.place(x=100, y=110)

    btn_seg = tk.Button(ventana_segmentacion, text="Segmentar", background="lightblue", command=segmentar)
    btn_seg.place(x=20, y=155)

    btn_recorte = tk.Button(ventana_segmentacion, text="Recortar", background="lightblue", command=recortar)
    btn_recorte.place(x=20, y=200)

    btn_eti = tk.Button(ventana_segmentacion, text="Etiquetar", background="lightblue", command=etiquetar)
    btn_eti.place(x=20, y=245)

    btnseg_limp = tk.Button(ventana_segmentacion, text="Limpiar ventana", background="yellowgreen", command=limpiar)
    btnseg_limp.place(x=20, y=290)

    # Crear un canvas que ocupe toda la ventana para simular una línea de separación
    canvas_linea = tk.Canvas(ventana_segmentacion, width=1, height=screen_height)
    canvas_linea.place(x=180)
    canvas_linea.create_line(1 / 2, 0, 1 / 2, screen_height, fill='black', width=5)

    # Control deslizante para ajustar alpha: cantidad de puntos
    alpha_label = tk.Label(ventana_segmentacion, text='Número de puntos:', background="white")
    alpha_label.place(x=20, y=340)
    alpha = tk.IntVar()
    alpha.set(10)  # Establece el valor inicial en 10
    alpha_slider = tk.Scale(ventana_segmentacion, from_=2, to=50, resolution=2, orient=tk.HORIZONTAL, variable=alpha,
                            background="lightgrey")
    alpha_slider.place(x=20, y=365)

    # Control deslizante para ajustar beta
    beta_label = tk.Label(ventana_segmentacion, text='Elasticidad:', background="white")
    beta_label.place(x=20, y=420)
    beta = tk.DoubleVar()
    beta.set(1)  # Establece el valor inicial en 1
    beta_slider = tk.Scale(ventana_segmentacion, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, variable=beta,
                           background="lightgrey")
    beta_slider.place(x=20, y=445)

    # Control deslizante para ajustar gamma
    gamma_label = tk.Label(ventana_segmentacion, text='Límite:', background="white")
    gamma_label.place(x=20, y=500)
    gamma = tk.DoubleVar()
    gamma.set(0.1)  # Establece el valor inicial en 0.1
    gamma_slider = tk.Scale(ventana_segmentacion, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, variable=gamma,
                            background="lightgrey")
    gamma_slider.place(x=20, y=525)

    # Interfaz para la barra de menús
    barra_menus = tk.Menu(ventana_segmentacion)
    menu_archivo = tk.Menu(barra_menus, tearoff=False)
    menu_acerca = tk.Menu(barra_menus, tearoff=False)
    archivos = tk.Menu(menu_archivo, tearoff=False)
    acerca = tk.Menu(menu_acerca, tearoff=False)

    menu_archivo.add_command(label="Nuevo", command=cargar_para_seg)
    menu_archivo.add_cascade(label="Guardar", menu=archivos)
    archivos.add_command(label="Imagen segmentada", command=guardar_seg)

    menu_archivo.add_command(label="Salir", accelerator="Esc", command=ventana_segmentacion.destroy)
    menu_acerca.add_cascade(label="Instrucciones", menu=acerca)
    acerca.add_command(label="Pasos a seguir para segmentar una imagen", command=instrucciones)
    barra_menus.add_cascade(menu=menu_archivo, label="Archivo")
    barra_menus.add_cascade(menu=menu_acerca, label="Ayuda")

    ventana_segmentacion.config(menu=barra_menus)
    ventana_segmentacion.bind("<Escape>", exit)

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- INTERFAZ GRÁFICA -----------------------------------------------------
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()
screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)
root = tk.Tk()
root.title( "App" )
root.geometry(str(int(screen_width/1.3))+"x"+str(int(screen_height/1.3)))
root.state('zoomed')
root.configure(bg='white')

# ----------- BOTONES -------------
btn_load = tk.Button(root, text="Cargar imagen", background="springgreen", command=load)
btn_load.place(x=20, y=20)

btn_filt = tk.Button(root, text="Filtrar imagen", background="springgreen", command=filtrado)
btn_filt.place(x=20, y=65)

btn_suav = tk.Button(root, text="Suavizar imagen", background="springgreen", command=suavizado)
btn_suav.place(x=20, y=110)

btn_bordes = tk.Button(root, text="Detectar bordes", background="springgreen", command=det_sobel)
btn_bordes.place(x=20, y=155)

btn_ventana = tk.Button(root, text="Segmentador", background="lightblue", command=abrir_ventana_segmentacion)
btn_ventana.place(x=20, y=200)

btn_destroy = tk.Button(root, text="Limpiar ventana", background="yellowgreen", command=destroy)
btn_destroy.place(x=20, y=245)

# Crear un canvas que ocupe toda la ventana para simular una línea de separación
canvas_linea = tk.Canvas(root, width=1, height=screen_height)
canvas_linea.place(x=180)
canvas_linea.create_line(1 / 2, 0, 1 / 2, screen_height, fill='black', width=5)

# Interfaz para la barra de menús
barra_menus = tk.Menu()
menu_archivo = tk.Menu(barra_menus, tearoff=False)
menu_opciones = tk.Menu(barra_menus, tearoff=False)
menu_acerca = tk.Menu(barra_menus, tearoff=False)
archivos = tk.Menu(menu_archivo, tearoff=False)
opciones = tk.Menu(menu_opciones, tearoff=False)
acerca = tk.Menu(menu_acerca, tearoff=False)

menu_archivo.add_command(label="Nuevo", command=load)
menu_archivo.add_cascade(label="Guardar", menu=archivos)
archivos.add_command(label="Imagen original", command=guardar_orig)
archivos.add_command(label="Histograma", command=guardar_hist)
archivos.add_command(label="Imagen filtrada", command=guardar_filt)
archivos.add_command(label="Imagen suavizada", command=guardar_suave)
archivos.add_command(label="Imagen bordes", command=guardar_borde)

menu_archivo.add_command(label="Salir", accelerator="Esc", command=root.destroy)
menu_opciones.add_command(label="Ver Histograma", command=mostrar_histograma)
menu_acerca.add_cascade(label="Instrucciones", menu=acerca)
acerca.add_command(label="Debe limpiar la ventana si quiere cargar una nueva imagen.")
barra_menus.add_cascade(menu=menu_archivo, label="Archivo")
barra_menus.add_cascade(menu=menu_opciones, label="Opciones")
barra_menus.add_cascade(menu=menu_acerca, label="Acerca de")

root.config(menu=barra_menus)
root.bind("<Escape>", exit)

root.update()
root.mainloop()