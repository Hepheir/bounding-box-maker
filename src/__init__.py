from __future__ import annotations

import dataclasses
import glob
import os
import typing


import numpy as np
import pandas as pd
import cv2


KEY_BACKSPACE = 127
KEY_SPACE = 32
KEY_ESC = 27
KEY_LEFT = 63234
KEY_RIGHT = 63235


class FileNameLoader:
    def __init__(self, dir: str, ext: str = 'jpg') -> None:
        self.dir = dir
        self.ext = ext
        self._ptr = 0
        self.filenames = sorted(self._filenames(), key=self._filename_indexing)

    def _filenames(self) -> typing.List[str]:
        return glob.glob('*.'+self.ext, root_dir=self.dir)

    def _filename_indexing(self, filename: str) -> int:
        return int(os.path.splitext(filename)[0])

    def _filename_abspath(self, filename: str) -> str:
        return os.path.join(self.dir, filename)

    def prev(self) -> None:
        self._ptr -= 1

    def next(self) -> None:
        self._ptr += 1

    def get(self) -> str:
        return self._filename_abspath(self.filenames[self._ptr])

    def get_id(self) -> int:
        return self._ptr


@dataclasses.dataclass
class Coordinates:
    x: int
    y: int


@dataclasses.dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    @classmethod
    def of(cls, c1: Coordinates, c2: Coordinates) -> BoundingBox:
        x = min(c1.x, c2.x)
        y = min(c1.y, c2.y)
        w = abs(c1.x - c2.x)
        h = abs(c1.y - c2.y)
        return BoundingBox(x, y, w, h)

    def __iter__(self) -> typing.Iterator[int]:
        yield self.x
        yield self.y
        yield self.w
        yield self.h


class CsvManager:
    def __init__(self, filename: str = 'bbox_data.csv') -> None:
        self.filename = filename
        self.columns = ['filename', 'x','y','w','h']
        self.data: typing.Dict[str, BoundingBox] = {}
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.filename):
            return
        df = pd.read_csv(self.filename)
        for i in range(len(df)):
            _, filename, x, y, w, h = iter(df.loc[i])
            self.data[filename] = BoundingBox(x,y,w,h)

    def save(self) -> None:
        data = [[key, *iter(value)] for key, value in self.data.items()]
        df = pd.DataFrame(data, columns=self.columns)
        df.to_csv(self.filename)

    def find(self, filename: str) -> typing.Optional[BoundingBox]:
        return self.data.get(filename, None)

    def set(self, filename: str, bbox: BoundingBox) -> None:
        self.data[filename] = bbox


class WindowHandler:
    def __init__(self, winname: str) -> None:
        self.bg: np.ndarray = np.zeros((100, 100))
        self.out: np.ndarray = self.bg
        self.winname: str = winname

    def run(self, strict_mode: bool = False) -> None:
        self.render()
        if strict_mode:
            self.loop()
        else:
            try:
                self.loop()
            except Exception as e:
                print(e)
                self.on_error()

    def loop(self) -> None:
        while True:
            if (key := cv2.waitKeyEx(10)) != -1:
                self.on_key_press(key)


    def set_background(self, image: np.ndarray) -> None:
        self.bg = image
        self.render()

    def render(self) -> None:
        cv2.imshow(self.winname, self.out)
        cv2.setMouseCallback(self.winname, self.mouse_event, self.out)

    def mouse_event(self, event:int, x: int, y: int, *args) -> None:
        coordinates = Coordinates(x, y)
        if event == cv2.EVENT_MOUSEMOVE:
            self.on_mouse_move(coordinates)
        if event == cv2.EVENT_FLAG_LBUTTON:
            self.on_mouse_click(coordinates)

    def on_mouse_move(self, coordinates: Coordinates) -> None:
        self.render()

    def on_mouse_click(self, coordinates: Coordinates) -> None:
        self.render()

    def on_key_press(self, key: int) -> None:
        self.render()

    def on_error(self) -> None:
        pass


class Labeler(WindowHandler):
    def __init__(self, data_dir: str, winname: str = 'image labeling') -> None:
        super().__init__(winname)
        self.mouse: Coordinates = Coordinates(0, 0)
        self.saved_mouse: typing.Optional[Coordinates] = None
        self.filename_loader = FileNameLoader(data_dir)
        self.csvManager = CsvManager()

    def run(self, strict_mode: bool = False) -> None:
        self.load_image()
        return super().run(strict_mode)

    def render(self) -> None:
        self.out = self.bg.copy()
        if self.csvManager.find(self.filename_loader.get()) is not None:
            self.draw_old_bounding_box()
        if self.saved_mouse is not None:
            self.draw_new_bounding_box()
        self.draw_cursor()
        return super().render()

    def draw_cursor(self) -> None:
        color = (0, 0, 255)
        thickness = 1
        h, w = self.out.shape[:2]
        cv2.line(self.out, (self.mouse.x, 0), (self.mouse.x, h), color, thickness)
        cv2.line(self.out, (0, self.mouse.y), (w, self.mouse.y), color, thickness)

    def draw_new_bounding_box(self) -> None:
        color = (0, 0, 255)
        thickness = 1
        cv2.rectangle(self.out, (self.saved_mouse.x, self.saved_mouse.y), (self.mouse.x, self.mouse.y), color, thickness)

    def draw_old_bounding_box(self) -> None:
        color = (0, 255, 0)
        thickness = 1
        bbox = self.csvManager.find(self.filename_loader.get())
        cv2.rectangle(self.out, (bbox.x, bbox.y), (bbox.x+bbox.w, bbox.y+bbox.h), color, thickness)

    def on_key_press(self, key: int) -> None:
        if key == KEY_SPACE:
            self.save_mouse_coordinates()
        if key == KEY_BACKSPACE:
            self.drop_mouse_coordinates()
        if key == KEY_LEFT:
            self.load_prev_image()
        if key == KEY_RIGHT:
            self.load_next_image()
        if key == KEY_ESC:
            raise StopIteration()
        super().on_key_press(key)

    def on_mouse_move(self, coordinates: Coordinates) -> None:
        self.mouse = coordinates
        return super().on_mouse_move(coordinates)

    def save_mouse_coordinates(self) -> None:
        if self.saved_mouse is None:
            self.saved_mouse = self.mouse
        else:
            self.save_bounding_box()
            self.drop_mouse_coordinates()

    def save_bounding_box(self) -> None:
        bbox = BoundingBox.of(self.saved_mouse, self.mouse)
        self.csvManager.set(self.filename_loader.get(), bbox)
        self.csvManager.save()

    def drop_mouse_coordinates(self) -> None:
        self.saved_mouse = None

    def load_prev_image(self) -> None:
        self.filename_loader.prev()
        self.load_image()

    def load_next_image(self) -> None:
        self.filename_loader.next()
        self.load_image()

    def load_image(self) -> None:
        self.set_background(cv2.imread(self.filename_loader.get()))


if __name__ == '__main__':
    Labeler('/Users/hepheir/GitHub/smu-cclab/WSPDUS/data/Products10k/train').run(True)
