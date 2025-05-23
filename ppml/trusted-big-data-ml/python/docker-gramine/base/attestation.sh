#!/bin/bash
set -x

# Attestation
if [ -z "$ATTESTATION" ]; then
    echo "[INFO] Attestation is disabled!"
    ATTESTATION="false"
elif [ "$ATTESTATION" = "true" ]; then
  echo "[INFO] Attestation is enabled!"
  # Build ATTESTATION_COMMAND
  if [ -z "$ATTESTATION_URL" ]; then
    echo "[ERROR] Attestation is enabled, but ATTESTATION_URL is empty!"
    echo "[INFO] PPML Application Exit!"
    exit 1
  fi
  if [ -z "$ATTESTATION_TYPE" -o "$ATTESTATION_TYPE" ==  "EHSMAttestationService" ]; then
    if [ -z "$APP_ID" ]; then
      echo "[ERROR] Attestation is enabled, but APP_ID is empty!"
      echo "[INFO] PPML Application Exit!"
      exit 1
    fi
    if [ -z "$API_KEY" ]; then
      echo "[ERROR] Attestation is enabled, but API_KEY is empty!"
      echo "[INFO] PPML Application Exit!"
      exit 1
    fi
  fi
  ATTESTATION_COMMAND="java -Xmx1g -cp $BIGDL_HOME/jars/*:$SPARK_HOME/conf/:$SPARK_HOME/jars/* com.intel.analytics.bigdl.ppml.attestation.AttestationCLI -u ${ATTESTATION_URL} -i ${APP_ID} -k ${API_KEY}"
  if [ -n "$ATTESTATION_CHALLENGE" ]; then
    ATTESTATION_COMMAND="${ATTESTATION_COMMAND} -c ${ATTESTATION_CHALLENGE}"
  fi
  if [ -n "$ATTESTATION_POLICYID" ]; then
    ATTESTATION_COMMAND="${ATTESTATION_COMMAND} -o ${ATTESTATION_POLICYID}"
  fi
  if [ -n "$ATTESTATION_TYPE" ]; then
    ATTESTATION_COMMAND="${ATTESTATION_COMMAND} -t ${ATTESTATION_TYPE}"
  fi
  echo $ATTESTATION_COMMAND > temp_command_file
  echo 'if [ $? -gt 0 ]; then ' >> temp_command_file
  echo '  exit 1' >> temp_command_file
  echo 'fi' >> temp_command_file
fi
