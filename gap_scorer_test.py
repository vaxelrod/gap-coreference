#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from gap_scorer import run_scorer


class TestScorer(unittest.TestCase):

  def test_correct_system_output_validation(self):
    # Two examples (one with Name A coref and one with Name B coref) are
    # correctly labeled in system output.
    scorecard = run_scorer('testdata/gold-validation.tsv',
                           'testdata/correct-validation.tsv')

    self.assertEqual(
        scorecard, 'Overall recall: 100.0 precision: 100.0 f1: 100.0\n'
        '\t\ttp 2\tfp 0\n'
        '\t\tfn 0\ttn 2\n'
        'Masculine recall: 100.0 precision: 100.0 f1: 100.0\n'
        '\t\ttp 1\tfp 0\n'
        '\t\tfn 0\ttn 1\n'
        'Feminine recall: 100.0 precision: 100.0 f1: 100.0\n'
        '\t\ttp 1\tfp 0\n'
        '\t\tfn 0\ttn 1\n'
        'Bias (F/M): 1.00\n')

  def test_correct_system_output_development(self):
    # Two examples (one with Name B coref and one with Neither name coref) are
    # correctly labeled in system output.
    scorecard = run_scorer('testdata/gold-development.tsv',
                           'testdata/correct-development.tsv')
    self.assertEqual(
        scorecard, 'Overall recall: 100.0 precision: 100.0 f1: 100.0\n'
        '\t\ttp 3\tfp 0\n'
        '\t\tfn 0\ttn 5\n'
        'Masculine recall: 100.0 precision: 100.0 f1: 100.0\n'
        '\t\ttp 1\tfp 0\n'
        '\t\tfn 0\ttn 3\n'
        'Feminine recall: 100.0 precision: 100.0 f1: 100.0\n'
        '\t\ttp 2\tfp 0\n'
        '\t\tfn 0\ttn 2\n'
        'Bias (F/M): 1.00\n')

  def test_incorrect_dataset_development(self):
    # All seemingly 'missing' IDs are scored as false negatives, e.g. if the
    # gold and system are from different datasets.
    scorecard = run_scorer('testdata/gold-development.tsv',
                           'testdata/correct-validation.tsv')
    self.assertEqual(
        scorecard, 'Overall recall: 0.0 precision: 0.0 f1: 0.0\n'
        '\t\ttp 0\tfp 0\n'
        '\t\tfn 8\ttn 0\n'
        'Masculine recall: 0.0 precision: 0.0 f1: 0.0\n'
        '\t\ttp 0\tfp 0\n'
        '\t\tfn 4\ttn 0\n'
        'Feminine recall: 0.0 precision: 0.0 f1: 0.0\n'
        '\t\ttp 0\tfp 0\n'
        '\t\tfn 4\ttn 0\n'
        'Bias (F/M): -\n')

  def test_incorrect_dataset_validation(self):
    scorecard = run_scorer('testdata/gold-validation.tsv',
                           'testdata/correct-development.tsv')
    self.assertEqual(
        scorecard, 'Overall recall: 0.0 precision: 0.0 f1: 0.0\n'
        '\t\ttp 0\tfp 0\n'
        '\t\tfn 4\ttn 0\n'
        'Masculine recall: 0.0 precision: 0.0 f1: 0.0\n'
        '\t\ttp 0\tfp 0\n'
        '\t\tfn 2\ttn 0\n'
        'Feminine recall: 0.0 precision: 0.0 f1: 0.0\n'
        '\t\ttp 0\tfp 0\n'
        '\t\tfn 2\ttn 0\n'
        'Bias (F/M): -\n')

  def test_incorrect_feminine_system_output(self):
    # Feminine example is incorrect in system output (Name A coref is called
    # Named B coref).
    scorecard = run_scorer('testdata/gold-validation.tsv',
                           'testdata/feminine-incorrect-validation.tsv')
    self.assertEqual(
        scorecard, 'Overall recall: 50.0 precision: 50.0 f1: 50.0\n'
        '\t\ttp 1\tfp 1\n'
        '\t\tfn 1\ttn 1\n'
        'Masculine recall: 100.0 precision: 100.0 f1: 100.0\n'
        '\t\ttp 1\tfp 0\n'
        '\t\tfn 0\ttn 1\n'
        'Feminine recall: 0.0 precision: 0.0 f1: 0.0\n'
        '\t\ttp 0\tfp 1\n'
        '\t\tfn 1\ttn 0\n'
        'Bias (F/M): -\n')

  def test_incorrect_masculine_system_output(self):
    # Masculine example is incorrect in system output (Name B coref is called
    # Neither name coref).
    scorecard = run_scorer('testdata/gold-validation.tsv',
                           'testdata/masculine-incorrect-validation.tsv')
    self.assertEqual(
        scorecard, 'Overall recall: 50.0 precision: 100.0 f1: 66.7\n'
        '\t\ttp 1\tfp 0\n'
        '\t\tfn 1\ttn 2\n'
        'Masculine recall: 0.0 precision: 0.0 f1: 0.0\n'
        '\t\ttp 0\tfp 0\n'
        '\t\tfn 1\ttn 1\n'
        'Feminine recall: 100.0 precision: 100.0 f1: 100.0\n'
        '\t\ttp 1\tfp 0\n'
        '\t\tfn 0\ttn 1\n'
        'Bias (F/M): -\n')

  def test_bias(self):
    # One masculine example is incorrect in system output (Neither name coref is
    # called Name A coref). This yields a non-trivial bias value.
    scorecard = run_scorer('testdata/gold-development.tsv',
                           'testdata/incorrect-development.tsv')
    self.assertEqual(
        scorecard, 'Overall recall: 100.0 precision: 75.0 f1: 85.7\n'
        '\t\ttp 3\tfp 1\n'
        '\t\tfn 0\ttn 4\n'
        'Masculine recall: 100.0 precision: 50.0 f1: 66.7\n'
        '\t\ttp 1\tfp 1\n'
        '\t\tfn 0\ttn 2\n'
        'Feminine recall: 100.0 precision: 100.0 f1: 100.0\n'
        '\t\ttp 2\tfp 0\n'
        '\t\tfn 0\ttn 2\n'
        'Bias (F/M): 1.50\n')


if __name__ == '__main__':
  unittest.main()
