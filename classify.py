import multiprocessing as mp
import tkinter
from tkinter import filedialog as file_dlg
from tkinter.messagebox import showwarning

import cv2
import numpy as np
from PIL import Image as PILImage
from PIL import ImageTk as PILImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

IMAGE_SIZE = (64, 64)
labels = ['Type_1', 'Type_2', 'Type_3', 'Type_4', 'Type_5']
win_main = tkinter.Tk()
img_queue = mp.Queue(10)
result_queue = mp.Queue(10)
stop = mp.Event()


def classify_thread(in_queue, out_queue, stop_event):
    model = load_model('classification_model.h5')
    while not stop_event.is_set():
        try:
            img = in_queue.get(block=True, timeout=0.1)
            image = cv2.resize(img, IMAGE_SIZE)
            image_arr = img_to_array(image)
            image_arr = np.expand_dims(image_arr, axis=0)
            probs = model.predict(image_arr)[0]
            proba = max(probs)
            label = labels[np.argmax(probs)]
            out_queue.put((proba, label))
        except Exception as e:
            pass


def image_t0_tk(img):
    blue, green, red = cv2.split(cv2.resize(img, (256, 256)))
    img = cv2.merge((red, green, blue))
    image_pil = PILImage.fromarray(img)
    return PILImageTk.PhotoImage(image=image_pil)


classify_thr = mp.Process(target=classify_thread, args=(img_queue, result_queue, stop,))
label_image = image_t0_tk(np.zeros((256, 256, 3), dtype=np.uint8))


def on_closing():
    global stop
    stop.set()
    classify_thr.join()
    win_main.destroy()


def load_image():
    global label_image
    file_types = (
        ('Images', '*.jpg *jpeg *.png *.tiff'),
        ('All files', '*.*'),
    )
    file_name = file_dlg.askopenfilename(
        title='Choose image of defect',
        filetypes=file_types
    )
    if len(file_name) > 0:
        try:
            image = cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        except Exception as e:
            showwarning(
                title='Error!',
                message='Unsupported image file!'
            )
            return
        img_queue.put(image)
        proba, label = result_queue.get(block=True)
        label_image = image_t0_tk(image)
        img_label.config(image=label_image)
        txt_label.config(text='Defect:\n {}\nwith probability {:.3f}'.format(label, proba))
        win_main.update_idletasks()


if __name__ == '__main__':
    classify_thr.start()
    win_main.geometry('396x396')
    win_main.title('Defect classifier')
    start_btn = tkinter.Button(text='Classify', font='Helvetica 16 bold')
    start_btn.config(command=load_image)
    start_btn.place(x=286, y=10, width=100, height=30)
    img_label = tkinter.Label(win_main, width=256, height=256, borderwidth=5, relief="raised")
    img_label.config(image=label_image)
    img_label.place(x=20, y=20, width=256, height=256)
    txt_label = tkinter.Label(win_main, text='', borderwidth=2, relief="sunken", font='Helvetica 18 bold')
    txt_label.place(x=20, y=286, width=300, height=100)
    win_main.protocol('WM_DELETE_WINDOW', on_closing)
    win_main.mainloop()
