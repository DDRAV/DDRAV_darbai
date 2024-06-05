#Sukurkite klasę Timer, kuri gali paleisti, sustabdyti ir iš naujo nustatyti laikmatį
# bei nurodyti praėjusį laiką sekundėmis. Naudokite time python paketą.

import time

class Timer:
    start_time = None
    elapsed_time = 0
    running = False

    @staticmethod
    def paleisti():
        if not Timer.running:
            Timer.start_time = time.time()
            Timer.running = True
            print("Laikmatis paleistas.")
        else:
            print("Laikmatis jau paleistas.")

    @staticmethod
    def sustabdyti():
        if Timer.running:
            Timer.elapsed_time += time.time() - Timer.start_time
            print(f"Laikmatis sustojo po {Timer.elapsed_time:.2f} sekundziu")
            Timer.start_time = None
            Timer.running = False
            print("Laikmatis sustabdytas.")
        else:
            print("Laikmatis dar nebuvo paleistas.")

    @staticmethod
    def atstatyti():
        Timer.start_time = None
        Timer.elapsed_time = 0
        Timer.running = False
        print("Laikmatis atstatytas.")

    @staticmethod
    def gauti_praejusi_laika():
        if Timer.running:
            current_time = time.time() - Timer.start_time
            total_time = Timer.elapsed_time + current_time
            print(f"Nuo starto praejo: {total_time}")
        print(f"Laikmatis ne paleistas")


Timer.paleisti()
time.sleep(2)
Timer.sustabdyti()
Timer.gauti_praejusi_laika()

Timer.paleisti()
time.sleep(3)
Timer.sustabdyti()
Timer.gauti_praejusi_laika()

Timer.atstatyti()