import random

from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from Dialog import SelectDialog
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout

from algorithms import *


class SelectPage(FloatLayout):

    def __init__(self, config, algorithm,**kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.algorithm = algorithm

    @staticmethod
    def page_index(*args):
        App.get_running_app().screen_manager.current = "Index_page"
        App.get_running_app().screen_manager.transition.direction = 'right'

    @staticmethod
    def page_database(*args):
        App.get_running_app().screen_manager.current = "Database_page"
        App.get_running_app().screen_manager.transition.direction = 'left'
    
    def choose_algorithm(self, text):
        self.algorithm = str(text)
    
    def get_input(self, random_n, input_n, m, n, n_content):
        n_content = n_content.split(",")
        if random_n:
            potential_letter = list(x for x in range(1, int(m)))
            self.config.n = random.sample(potential_letter, int(n))
        elif input_n:
            if not n_content or len(n_content) < int(n):
                box = BoxLayout(orientation='vertical')  # 创建一个BoxLayout来包含Label和Button
                box.add_widget(Label(text='Please enter the correct value for n!'))
                button = Button(text='OK',size_hint=(0.25, 0.25),pos_hint={'center_x':0.5})  # 创建一个Button
                button.bind(on_release=lambda _: self.popup.dismiss())  # 当按钮被点击时，关闭Popup
                box.add_widget(button)
                self.popup = Popup(title='Input Required',
                    content=box,  # 使用包含Label和Button的BoxLayout作为Popup的内容
                    size_hint=(None, None), size=(500, 360))
                self.popup.open()
            self.config.n = n_content

        return self.config.n

    def run_algorithms(self, m, n, k, j, s):
        config = list(map(int, [m, n, k, j, s]))
        if self.algorithm == 'Greedy':
            # self.algorithm = 'Greedy'
            pass
        elif self.algorithm == 'GA':
            self.config.result = run_ga(config)
        elif self.algorithm == 'SA':
            self.config.result = run_sa(config)
        elif self.algorithm == 'PSO':
            self.config.result = run_pso(config)
        elif teself.algorithmxt == 'AFSA':
            self.config.result = run_afsa(config)
        elif self.algorithm == 'ACA':
            self.config.result = run_aca(config)

        return self.config.result

    def import_database(self):
        self.dismiss_popup()

    def dismiss_popup(self):
        self.popup.dismiss()