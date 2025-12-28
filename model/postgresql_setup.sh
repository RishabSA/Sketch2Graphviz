apt update
apt install -y postgresql-common ca-certificates
apt install -y postgresql postgresql-contrib
apt install -y git build-essential postgresql-server-dev-14

cd /tmp
git clone --branch v0.8.1 https://github.com/pgvector/pgvector.git
cd pgvector
make && make install


service postgresql start
service postgresql status
su - postgres
psql

CREATE ROLE root WITH LOGIN SUPERUSER;
CREATE DATABASE sketch2graphvizdb;
\c sketch2graphvizdb
CREATE EXTENSION IF NOT EXISTS vector;
\dx

\q

exit

su - postgres -c "psql -d sketch2graphvizdb -f /app/postgreSQL_data/sketch2graphvizdb.sql"