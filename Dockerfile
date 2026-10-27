# SPDX-License-Identifier: Apache-2.0
ARG PYTHON_IMAGE=python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d
FROM ghcr.io/astral-sh/uv:0.12.23@sha256:61d393e44e249f2e4b526b6c7ddcecce245946826e608e11c93ad4f5bba55b21 AS uv-tools
FROM ${PYTHON_IMAGE} AS development
COPY --from=uv-tools /uv /uvx /usr/local/bin/
RUN --mount=type=secret,id=ca,required=false,mode=0444 if [ -f /run/secrets/ca ]; then export SSL_CERT_FILE=/run/secrets/ca; fi; sed -i "s|http://deb.debian.org|https://deb.debian.org|g" /etc/apt/sources.list.d/debian.sources && apt-get -o Acquire::https::CaInfo="${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}" update && apt-get -o Acquire::https::CaInfo="${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}" install -y --no-install-recommends git curl ca-certificates dpkg-dev libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2 libpango-1.0-0 libcairo2 && rm -rf /var/lib/apt/lists/*
ARG TASK_VERSION=3.54.0
RUN --mount=type=secret,id=ca,required=false,mode=0444 set -eu; \
    if [ -f /run/secrets/ca ]; then export CURL_CA_BUNDLE=/run/secrets/ca; fi; \
    arch="$(dpkg --print-architecture)"; mkdir /tmp/task; cd /tmp/task; \
    curl --connect-timeout 15 --max-time 120 -fsSLO "https://github.com/go-task/task/releases/download/v${TASK_VERSION}/task_linux_${arch}.tar.gz"; \
    curl --connect-timeout 15 --max-time 120 -fsSLO "https://github.com/go-task/task/releases/download/v${TASK_VERSION}/task_checksums.txt"; \
    sha256sum --ignore-missing -c task_checksums.txt; \
    tar -xzf "task_linux_${arch}.tar.gz" task; install task /usr/local/bin/task; rm -rf /tmp/task
ENV UV_PROJECT_ENVIRONMENT=/opt/mujoco-venv UV_CACHE_DIR=/opt/mujoco-cache UV_PYTHON_DOWNLOADS=never UV_SYSTEM_CERTS=true PIP_CACHE_DIR=/opt/mujoco-cache/pip
RUN mkdir /opt/mujoco-venv /opt/mujoco-cache && chmod 1777 /opt/mujoco-venv /opt/mujoco-cache
WORKDIR /workspace/mujoco
CMD ["sleep", "infinity"]

FROM ${PYTHON_IMAGE} AS runtime
COPY dist/wheelhouse /tmp/wheelhouse
RUN pip install --no-index --no-deps /tmp/wheelhouse/*.whl && rm -rf /tmp/wheelhouse
ARG VERSION=0.0.0
LABEL org.opencontainers.image.title="livingproof-sim-mujoco" org.opencontainers.image.version=$VERSION org.opencontainers.image.licenses="Apache-2.0" org.opencontainers.image.source="https://github.com/Living-Proof-Labs/livingproof-sim-mujoco"
USER 65532:65532
WORKDIR /tmp
ENTRYPOINT ["livingproof-sim-mujoco"]

FROM runtime AS runtime-test
USER root
COPY dist/test-wheelhouse /tmp/test-wheelhouse
RUN pip install --no-index --no-deps /tmp/test-wheelhouse/*.whl && rm -rf /tmp/test-wheelhouse
COPY tests/test_cli.py /tests/test_cli.py
RUN chmod -R a+rX /tests
ENV INSTALLED_PRODUCT=1
WORKDIR /tmp
ENTRYPOINT ["python", "-m", "pytest", "/tests", "-p", "no:cacheprovider"]
