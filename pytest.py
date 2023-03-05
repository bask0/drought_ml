

cmd_1 = 'srun python3 project/clitest.py fit -c project/config/tcn.yaml  --trainer.logger.save_dir ./logs/tcn_test --trainer.logger.version trial_0_2023-02-02_22-26-44'
cmd_2 = 'srun python3 project/clitest.py predict -c project/config/tcn.yaml  --trainer.logger.save_dir ./logs/tcn_test --trainer.logger.version trial_0_2023-02-02_22-26-44'

def get_str():
    s = \
    """
    while getopts 'tfp' OPTION; do
    case "$OPTION" in
        t)
        {0}

        wait

        {1}
        ;;
        f)
        {0}
        ;;
        p)
        ;;
        f)
        {1}
        ;;
        ?)
        echo "script usage: $(basename \$0) [-t] [-f] [-p]" >&2
        exit 1
        ;;
    esac
    done
    """
    return s

print(get_str().format(cmd_1, cmd_2))
