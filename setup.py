import os
from glob import glob

from setuptools import find_packages, setup

package_name = "robot_vision"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # launch 파일 설치 설정
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        # model 디렉터리의 best.xml, best.bin 파일 설치 설정
        (os.path.join("share", package_name, "model"), glob("model/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="hojin",
    maintainer_email="hojin7710@daum.net",
    description="TODO: Package description",
    license="TODO: License declaration",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            # '실행할_명령어_이름 = 패키지명.파일명:실행함수' 형태
            "image_sub_node = robot_vision.image_sub_node:main",
        ],
    },
)
