.PHONY: torcs-procs torcs-ports kill-torcs clean-torcs

torcs-procs:
	@echo "TORCS processes:"
	@ps aux | grep -i '[t]orcs' || true

torcs-ports:
	@echo "TORCS/SCR ports:"
	@ss -ltnp | grep ':300' || true

.PHONY: kill-torcs

kill-torcs:
	@echo "Hard-killing all TORCS processes..."
	@pgrep -af 'torcs' || true
	@pgrep -f 'torcs' | xargs -r kill -9
	@sleep 0.5
	@rm -rf /tmp/torcs-*
	@echo "Remaining TORCS processes:"
	@pgrep -af 'torcs' || true
	@echo "Remaining TORCS ports:"
	@ss -ltnp | grep ':300' || true

clean-torcs: kill-torcs
	@echo "Removing temporary TORCS folders..."
	@rm -rf /tmp/torcs-*
	@echo "Remaining TORCS ports:"
	@ss -ltnp | grep ':300' || true