//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
// G4DNAMolecularReaction_ice.hh - Ice-specific molecular reaction with Arrhenius diffusion
//
// Author: Mathieu Karamitros (original)
// Modified for ice chemistry with Arrhenius diffusion coefficients
//
// This ice-specific version handles molecular reactions using position-dependent
// diffusion coefficients calculated via Arrhenius equation: D = D₀ × exp(-Eₐ/kT)
// where temperature varies with depth in ice. Overrides MakeReaction() to use
// ice-specific coefficients for OH (Eₐ=0.14 eV) and H (Eₐ=0.02 eV) molecules.
//

#pragma once

#include "G4DNAMolecularReaction.hh"

class G4MolecularConfiguration;
class G4Track;

/**
  * G4DNAMolecularReaction_ice: Ice-specific molecular reaction handler
  * 
  * Handles molecular reactions using position-dependent diffusion coefficients
  * calculated via Arrhenius equation. Temperature varies with depth in ice,
  * causing diffusion rates to change by orders of magnitude.
  * 
  * Arrhenius parameters:
  * - OH molecules: D₀ = 9×10⁻⁸ m²/s, Eₐ = 0.14 eV
  * - H molecules:  D₀ = 9×10⁻⁸ m²/s, Eₐ = 0.02 eV
  * - Other molecules: Use stored coefficients
  */
class G4DNAMolecularReaction_ice : public G4DNAMolecularReaction
{
public:
    // Constructors: Identical to base class
    G4DNAMolecularReaction_ice();
    explicit G4DNAMolecularReaction_ice(G4VDNAReactionModel*);
    ~G4DNAMolecularReaction_ice() override = default;
    G4DNAMolecularReaction_ice(const G4DNAMolecularReaction_ice& other) = delete;
    G4DNAMolecularReaction_ice& operator=(const G4DNAMolecularReaction_ice& other) = delete;

    // Main override: Uses Arrhenius coefficients for OH/H molecules
    std::unique_ptr<G4ITReactionChange> MakeReaction(const G4Track&, const G4Track&) override;

private:
    // Helper: Calculates position-dependent diffusion coefficients
    // Uses D = D₀ × exp(-Eₐ/kT) for OH/H, stored coefficients for others
    G4double GetArrheniusDiffusionCoefficient(const G4Track& track, const G4MolecularConfiguration* molConf);
};
