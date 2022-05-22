using Pkg
Pkg.add("MySQL")       # No need for a full PackageSpec here
Pkg.add("DBInterface")

using MySQL
using DBInterface
conn = DBInterface.connect(MySQL.Connection, "ec2blah.eu-west-2.compute.amazonaws.com",
                           "name", "password", db = "database")

