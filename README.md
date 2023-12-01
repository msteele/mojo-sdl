# Minimal SDL binding for Mojo 🔥
Very basic native Mojo bindings to SDL2. No Python is used at runtime.
A Python script to generate different colormaps for the demo is included.

## Usage
Requires SDL installed:
#### Ubuntu
```bash
apt install libsdl2-dev
```
#### Fedora
```bash
dnf install sdl2-devel
```
#### Mac OS
```bash
brew install sdl2
```


Copy `SDL.mojo` to your project or try the Mandelbrot demo ported from [Mojo examples](https://github.com/modularml/mojo/tree/main/examples/mandelbrot.mojo)

## Demo
Run
```bash
mojo mandelbrot.mojo
```
and zoom in/out with the mouse wheel.
