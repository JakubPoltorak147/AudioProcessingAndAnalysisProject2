from tkinter import Tk
from audio_app import AudioApp
import sys

def main():
    root = Tk()
    root.title("AudioApp")
    root.geometry("900x700")
    app = AudioApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
    sys.exit(0)


if __name__ == "__main__":
    main()
