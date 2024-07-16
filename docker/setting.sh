#!/bin/zsh

####
# python install
####
pyenv install 3.10.12
# デフォルトの python バージョンを 3.10.12 にする
pyenv global 3.10.12

####
# poetry install
# (poetry requires python>=3.7)
####
curl -sSL https://install.python-poetry.org | python3 - --version 1.5.1
# poetry で作る仮想環境をプロジェクト直下に生成するようにする
poetry config virtualenvs.in-project true
# プロジェクトディレクトリ project を作成して移動
cd $HOME/pres_lens
pyenv local 3.10.12
# 仮想環境の構築 (poetry をダウンロードした python と別のバージョンを使用する場合必須)
# TODO: nanikore
# python -m venv .venv
# pyproject.toml, poetry.lock をもとに module をインストール
poetry install
