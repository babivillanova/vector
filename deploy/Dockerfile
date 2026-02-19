FROM public.ecr.aws/lambda/python:3.12

# Install build tools for potrace compilation
RUN dnf install -y gcc make tar gzip && dnf clean all

# Compile potrace from source (not in Amazon Linux repos)
ARG POTRACE_VERSION=1.16
RUN curl -fsSL https://potrace.sourceforge.net/download/${POTRACE_VERSION}/potrace-${POTRACE_VERSION}.tar.gz \
      -o /tmp/potrace.tar.gz && \
    cd /tmp && tar xzf potrace.tar.gz && \
    cd potrace-${POTRACE_VERSION} && \
    ./configure --prefix=/usr/local && make && make install && \
    rm -rf /tmp/potrace*

# Install Python dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy application code
COPY fresh_vectorize_layered.py ${LAMBDA_TASK_ROOT}/
COPY api.py ${LAMBDA_TASK_ROOT}/
COPY handler.py ${LAMBDA_TASK_ROOT}/

CMD ["handler.handler"]
