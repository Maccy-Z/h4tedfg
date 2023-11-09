FROM ubuntu:jammy
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y git curl
RUN apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
RUN apt install nano

# Install custom python version for use with Julia
WORKDIR /home/active_phase
ENV HOME  /home/active_phase
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN curl https://pyenv.run | bash
RUN PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.11
RUN pyenv global 3.11

RUN pip3 install julia
RUN pip install numpy matplotlib

# Install Julia
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.3-linux-x86_64.tar.gz
RUN tar zxvf julia-1.9.3-linux-x86_64.tar.gz
ENV PATH $PATH:$HOME/julia-1.9.3/bin

# Integrate Julia with python
RUN julia -e 'using Pkg; Pkg.add("PyCall")'
RUN pip install julia

# Copy my files over
COPY ./AugmentedGaussianProcesses.jl $HOME/AugmentedGaussianProcesses.jl
# Precompile packages. Needs git repo
WORKDIR $HOME/AugmentedGaussianProcesses.jl
RUN  git config --global user.email "you@example.com"
RUN  git config --global user.name "Your Name"

RUN git init
RUN git add .
RUN git commit -m "Initial commit"

# Install packages
RUN julia install.jl
