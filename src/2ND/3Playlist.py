class Playlist:
    def __init__(self):
        self.songs = []

    def add_song(self, song_name: str):
        if song_name in self.songs:
            print(f"Daina {song_name} jau yra grojarastyje")
        else:
            self.songs.append(song_name)
            print(f"Daina {song_name} prideta prie grojarascio")

    def remove_song(self, song_name: str):
        if song_name in self.songs:
            self.songs.remove(song_name)
            print(f"Daina {song_name} pasalinta is grojarascio")
        else:
            print(f"Daina {song_name} nerasta grojarastyje")

    def show_playlist(self):
        if self.songs:
            print("Grojarastis:")
            for song in self.songs:
                print(f"- {song}")
        else:
            print("Grojarastis tuscias")


my_playlist = Playlist()

my_playlist.show_playlist()
my_playlist.add_song("Daina 1")
my_playlist.add_song("Daina 2")

my_playlist.show_playlist()
my_playlist.remove_song("Daina 1")
my_playlist.show_playlist()

my_playlist.remove_song("Daina 3")
my_playlist.add_song("Daina 3")
my_playlist.show_playlist()

my_playlist.remove_song("Daina 2")
my_playlist.show_playlist()
