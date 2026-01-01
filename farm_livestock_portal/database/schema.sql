CREATE DATABASE farm_db;
USE farm_db;

CREATE TABLE livestock (
    id INT AUTO_INCREMENT PRIMARY KEY,
    animal_tag VARCHAR(100) UNIQUE,
    animal_type VARCHAR(50),
    breed VARCHAR(50),
    age INT,
    health_status VARCHAR(100),
    purchase_date DATE
);
