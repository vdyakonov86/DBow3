FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive

# create a non-root user
ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ENV LANG=en_US.UTF-8

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && mkdir /home/$USERNAME/.config && chown $USER_UID:$USER_GID /home/$USERNAME/.config

# set up sudo privileges
RUN apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
    && chmod 0440 /etc/sudoers.d/$USERNAME

# add user to video group to allow access to webcam
RUN usermod --append --groups video $USERNAME

RUN apt-get install -y --no-install-recommends \
    vim \
    git \
    wget \
    curl \
    zip \
    unzip \
    python3-pip \
    x11-apps \
    # Autocomplete
    bash-completion \
    python3-argcomplete \
    cmake \
    g++ \
    # onnx_runtime_cpp deps
    gnupg2 \
    lsb-release \
    build-essential \
    software-properties-common \
    tmux 

RUN apt-get -y purge cmake && sudo pip install cmake==3.25.0

# --- OpenCV
# fix: OpenCV(4.9.0) /opencv/modules/highgui/src/window.cpp:1272: error: (-2:Unspecified error) The function is not implemented.
RUN apt-get install -y libgtk2.0-dev pkg-config

# download and unpack sources
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.9.0.zip \
    && unzip opencv.zip \
    && mv opencv-4.9.0 opencv \
    && rm opencv.zip

# build
RUN mkdir -p /opencv_build \
    && cd /opencv_build \
    && cmake -DBUILD_TESTS=OFF ../opencv \
    && make -j4

# install, remove source/build files
RUN cd /opencv_build \
    && sudo make install \
    && cd .. \
    && rm -rf /opencv /opencv_build
# --- End OpenCV


# extra deps for onnx_runtime_cpp build
RUN apt-get install -y libboost-dev libboost-serialization-dev

# Remove folder in order to ensure to run apt-get update before installing new package 
# (without this folder 'apt-get update' will not work)
RUN rm -rf /var/lib/apt/lists/*

COPY scripts/.bashrc /home/${USERNAME}/bashrc
RUN cat /home/${USERNAME}/bashrc >> /home/${USERNAME}/.bashrc && rm /home/${USERNAME}/bashrc

WORKDIR /fbow
# switch from root to user
USER $USERNAME

ENTRYPOINT ["/bin/bash"]