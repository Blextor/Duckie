source startvx.sh

sleep 2

xdpyinfo -display :99 >/dev/null 2>&1 && echo "In use" || echo "Free"

python3 groundtruth.py
