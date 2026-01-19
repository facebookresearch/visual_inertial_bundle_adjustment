/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <viba/common/Report.h>

#include <nlohmann/json.hpp>
#include <fstream>

namespace visual_inertial_ba {

void writeJsonReport(
    const std::filesystem::path& outputFilePath,
    const small_thing::Optimizer::Summary& summary) {
  nlohmann::json json;
  nlohmann::json jSummary = json["summary"];

  jSummary["initial_cost"] = summary.initialCost;
  jSummary["final_cost"] = summary.finalCost;
  jSummary["num_iterations"] = summary.numIterations;
  jSummary["num_troubled_seqs"] = summary.numTroubledSeqs;
  jSummary["largest_troubled_seq"] = summary.largestTroubledSeq;
  json["summary"] = jSummary;

  std::ofstream file(outputFilePath);
  file << json.dump(2);
}

} // namespace visual_inertial_ba
