apt-get update -qq
apt-get install -y -qq ca-certificates curl gnupg2 lsb-release > /dev/null
curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc | gpg --dearmor -o /etc/apt/trusted.gpg.d/postgresql.gpg
echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list

# Install PostgreSQL 18
apt-get update -qq
apt-get install -y -qq postgresql-18 postgresql-contrib-18 > /dev/null

service postgresql start
psql --version

# Install PGVector
sudo apt update && sudo apt install postgresql-18-pgvector

sudo -u postgres psql -c "CREATE USER root WITH SUPERUSER"
sudo -u postgres psql -c "CREATE DATABASE sketch2graphvizdb"

# Setup PGVector
sudo -u postgres psql -c "\c sketch2graphvizdb"

sudo -u postgres psql -c "CREATE EXTENSION IF NOT EXISTS vector;"
sudo -u postgres psql -c "\dx"

# Load sketch2graphvizdb data
psql -d sketch2graphvizdb -f sketch2graphvizdb.sql