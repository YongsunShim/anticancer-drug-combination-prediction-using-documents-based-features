#!/bin/sh
mvn clean
mvn compile
mvn exec:exec -Dexec.executable="java" -Dexec.args="-Xmx128G -Xms64G -classpath %classpath bike.snu.ac.kr.dcpl.anticancer_drug_combination_prediction_using_documents_based_features.Main" -Dexec.workingDirectory="target/classes"
