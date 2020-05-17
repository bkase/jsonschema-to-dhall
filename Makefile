BIN ?= cargo run

generate:
	$(BIN)
	patch -p1 < fix-automatic-union-patch.p1

install:
	dhall <<< '$(out)/definitions/commandStep/Type'

release:
	rm -rf out
	cp -r result/ out
	chmod -R 775 out
