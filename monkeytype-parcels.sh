#! /bin/bash
pytest --monkeytype-output="monkeytype.sqlite3" tests
cat monkeytype-modules.txt | xargs -n1 monkeytype stub