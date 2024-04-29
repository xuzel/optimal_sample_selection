import logging

from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup

from Dialog import DownloadDialog
from database.database import Database


class DatabasePage(FloatLayout):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.database = Database()

    @staticmethod
    def page_index(*args):
        App.get_running_app().screen_manager.current = "Index_page"
        App.get_running_app().screen_manager.transition.direction = 'right'

    @staticmethod
    def page_select(*args):
        App.get_running_app().screen_manager.current = "Select_page"
        App.get_running_app().screen_manager.transition.direction = 'right'

    @staticmethod
    def page_refilter(*args):
        App.get_running_app().screen_manager.current = "Refilter_page"
        App.get_running_app().screen_manager.transition.direction = 'left'

    def query_specified_data(self, m, n, k, j, s, times):
        format_data = self.format_data(m, n, k, j, s)
        output = self.database.find_many(format_data)
        format_output = self.format_output(output)
        return format_output

    def delete_specified_data(self, m, n, k, j, s, times):
        format_data = self.format_data(m, n, k, j, s)
        # outcomes = self.database.delete_one(format_data, times)
        outcomes = self.database.delete_many(format_data)
        output = self.database.find_many(format_data)
        format_output = self.format_output(output)
        # return outcomes + "\n" + format_outputrn
        return outcomes

    def format_data(self, m, n, k, j, s):
        return m + "-" + n + "-" + k + "-" + j + "-" + s

    def format_output(self, output):
        format_output = ''
        for item in output:
            format_output = format_output + item[0] + ':\n' + item[1] + "\n"
        return format_output

    def show_load(self):
        # content = LoadDialog(load=self._load,cancel=self.dismiss_popup,cwdir=os.getcwd())
        content = DownloadDialog(load=self._load, cancel=self.dismiss_popup, cwdir="./")
        self._popup = Popup(title="Download Database File", content=content, size_hint=(.9, .9))
        self._popup.open()

    def _load(self, path, file):
        try:
            file_path =file[0] + '\DB.txt'
        except:
            file_path = path + '\DB.txt'
            print("path: " + file_path)
        print("path: " + file_path)
        with open(file_path, "w", encoding='utf-8') as file:
            file.write(self.query_all_data())
        self.dismiss_popup()

    def dismiss_popup(self):
        self._popup.dismiss()

    def query_all_data(self):
        output = self.database.find_all()
        format_output = self.format_output(output)
        return format_output
