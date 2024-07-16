#!/usr/bin/env zsh

HOME="/workspace"
# container が build される直前に実行される処理を記述
# default shell を指定
chsh -s /bin/zsh
# shell の再起動
source /workspace/.zshrc
# download pyenv packege
git clone https://github.com/pyenv/pyenv.git $HOME/.pyenv

zsh
