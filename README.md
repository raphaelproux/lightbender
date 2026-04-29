If you don't have VS code with C++ build tools, or just the C++ build tools installed, you can use the MINGW compiler:
```
scoop install mingw
rustup toolchain install stable-x86_64-pc-windows-gnu
rustup default stable-x86_64-pc-windows-gnu

```

It might be necessary to set up an env var (to be tested):
```$env:CARGO_BUILD_TARGET = "x86_64-pc-windows-gnu"```