DIR="external"
git clone https://github.com/Microsoft/vcpkg.git "$DIR/vcpkg"
    cd "$DIR/vcpkg"
    ./bootstrap-vcpkg.sh
    ./vcpkg integrate install
    ./vcpkg install boost
    ./vcpkg install eigen3
