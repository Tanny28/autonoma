-- AUTONOMA PostgreSQL initialization
-- Creates application databases on first container start.
-- The default POSTGRES_DB (autonoma) is created automatically by the postgres image.

CREATE DATABASE mlflow;
CREATE DATABASE autonoma_app;
