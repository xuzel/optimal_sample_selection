from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image


class MainScreen(BoxLayout):
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = 10
        self.spacing = 10

        # Add your logo
        self.add_widget(Image(source='logo.png', size_hint=(1, 0.5)))

        # Version and Title
        self.add_widget(Label(text='Optimal Sample Selection\nv.4.2.1', size_hint=(1, 0.1)))

        # Select Button
        self.select_button = Button(text='Select', size_hint=(1, 0.15))
        self.add_widget(self.select_button)

        # Database Button
        self.database_button = Button(text='Database', size_hint=(1, 0.15))
        self.add_widget(self.database_button)


class OSSApp(App):
    def build(self):
        return MainScreen()


if __name__ == '__main__':
    OSSApp().run()
