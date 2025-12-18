cat > Dockerfile <<'EOF'
FROM nvcr.io/nvidia/tensorflow:25.01-tf2-py3

WORKDIR /workspace

RUN python -m pip install -U pip setuptools wheel && \
    python -m pip install sktime
EOF
