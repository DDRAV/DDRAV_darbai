class FitnessTracker:
    def __init__(self, user_name: str):
        self.user_name = user_name
        self._steps = 0

    def add_steps(self, steps: int):
        self._steps += steps
        self._check_goal()

    def reset_steps(self):
        self._steps = 0
        print(f"{self.user_name} jusu zingsniu skaicius nunulintas")
        self._check_goal()

    def _check_goal(self):
        if self._steps >= 10000:
            print(f"Sveikaname {self.user_name}, minimalus žingsniu kiekis šiandienai pasiektas")
        elif self._steps < 10000:
            rez = 10000 - self._steps
            print(f"{self.user_name} jums liko {rez} zingsniu iki tikslo, dabar privaiksciojot {self._steps} zingsniu")
    #nerasiau get_steps funkcijos nes _check_goal funkcijoi jei nesurinktas zingsniu skaicius grazinama kiek truksta zingsiniu ir kiek privaiksciota

track1 = FitnessTracker(user_name="Jonas")
track2 = FitnessTracker(user_name="Onute")

track1._check_goal()
track1.add_steps(11000)
track1.reset_steps()
track1.add_steps(10000)
track2._check_goal()
track2.add_steps(7500)

