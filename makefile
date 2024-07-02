include NLIB_CROSS

all: ${EXE_FILE}

${EXE_FILE}:
	${CC} -o ${EXE_FILE} ${C_FILE} ${CFLAGS}

clean:
	rm -rf *.adx *.log *.map
