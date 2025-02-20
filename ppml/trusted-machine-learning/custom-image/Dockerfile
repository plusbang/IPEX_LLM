ARG MACHINE_LEARNING_BASE_IMAGE_NAME
ARG MACHINE_LEARNING_BASE_IMAGE_TAG

# stage 1. generate SGX secrets and prepare KMS env
FROM $MACHINE_LEARNING_BASE_IMAGE_NAME:$MACHINE_LEARNING_BASE_IMAGE_TAG as temp

ARG SGX_MEM_SIZE
ARG SGX_LOG_LEVEL
ARG ENABLE_DCAP_ATTESTATION

ADD ./enclave-key.pem /root/.config/gramine/enclave-key.pem
ADD ./make-sgx.sh /ppml/make-sgx.sh

# 1.1 make SGX and sign in a temp image
RUN cd /ppml && \
    chmod a+x make-sgx.sh && \
    ./make-sgx.sh

# stage 2. copy sign etc. secrets from temp into target image and generate AS secrets
FROM $MACHINE_LEARNING_BASE_IMAGE_NAME:$MACHINE_LEARNING_BASE_IMAGE_TAG

# 2.1 copy sign etc. secrets from temp into target image
COPY --from=temp /ppml/bash.manifest.sgx /ppml/bash.manifest.sgx
COPY --from=temp /ppml/bash.sig /ppml/bash.sig
COPY --from=temp /ppml/bash.manifest /ppml/bash.manifest

# 2.2 output mr_enclave and mr_signer to customer
RUN cd /ppml && \
    gramine-sgx-get-token --output bash.token --sig bash.sig

WORKDIR /ppml

ENTRYPOINT [ "/opt/entrypoint.sh" ]
