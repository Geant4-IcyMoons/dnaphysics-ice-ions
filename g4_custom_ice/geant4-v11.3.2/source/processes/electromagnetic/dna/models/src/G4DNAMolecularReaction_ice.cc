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
// G4DNAMolecularReaction_ice.cc - Ice-specific molecular reaction with Arrhenius diffusion
// ---------------------------------------------------------------------
//  G4DNAMolecularReaction_ice.cc
//  Customized by G. Yoffe for ice (solid H2O) chemistry
//
//  Implements molecular reactions in ice using Arrhenius diffusion
//  coefficients. Differs from the original by using temperature-dependent
//  coefficients and spatial logic for ice, not liquid water.
// ---------------------------------------------------------------------
//
// Author: Mathieu Karamitros (original)
// Modified for ice chemistry with Arrhenius diffusion coefficients
//
// IMPLEMENTATION OVERVIEW:
// This file implements position-dependent molecular reactions for Europa's ice chemistry.
// The key innovation is replacing fixed diffusion coefficients with dynamic Arrhenius
// calculations that account for Europa's dramatic temperature variations.
//
// PHYSICS RATIONALE:
// Europa's ice spans from ~200K (warm, near-surface) to ~100K (cold, deep interior).
// Molecular diffusion rates follow Arrhenius behavior: D = D₀ × exp(-Eₐ/kT)
// At 200K vs 100K, OH diffusion changes by ~10³ orders of magnitude!
// Vanilla Geant4 ignores this, leading to unphysical chemistry results.
//
// ALGORITHMIC CHANGES FROM VANILLA:
// 1. MakeReaction(): Uses GetArrheniusDiffusionCoefficient() instead of stored values
// 2. Reaction site calculation weighted by actual diffusion rates at current position
// 3. Temperature lookup integrated into every reaction event
// 4. Maintains compatibility with non-OH/H molecules using fallback coefficients
//

#include "G4DNAMolecularReaction_ice.hh"
#include "G4DNAMolecularReactionTable.hh"
#include "G4VDNAReactionModel.hh"
#include "G4MolecularConfiguration.hh"
#include "G4Molecule.hh"
#include "G4MoleculeFinder.hh"
#include "G4ITReactionChange.hh"
#include "G4ITReaction.hh"
#include "G4ITTrackHolder.hh"

// Ice-specific includes for Arrhenius diffusion coefficients
#include "ArrheniusConstants_ice.hh"
#include "G4OH.hh"
#include "G4Hydrogen.hh"

// Helper function to get ice temperature at position (from G4DNABrownianTransportation_ice.cc)
extern G4double GetIceTemperatureAtPosition(const G4ThreeVector& position);

// CONSTRUCTOR IMPLEMENTATIONS
// These are identical to vanilla - ice behavior comes from method overrides, not initialization
G4DNAMolecularReaction_ice::G4DNAMolecularReaction_ice()
    : G4DNAMolecularReaction()
{
}

G4DNAMolecularReaction_ice::G4DNAMolecularReaction_ice(G4VDNAReactionModel* pReactionModel)
    : G4DNAMolecularReaction(pReactionModel)
{
}

// ICE-SPECIFIC ARRHENIUS HELPER FUNCTION
// VANILLA DIFFERENCE: Vanilla class has no equivalent function
// This is the core innovation that enables position-dependent chemistry
G4double G4DNAMolecularReaction_ice::GetArrheniusDiffusionCoefficient(const G4Track& track, const G4MolecularConfiguration* molConf) {
    // Step 1: Get current position and temperature
    // VANILLA: Never considers position or temperature
    G4ThreeVector position = track.GetPosition();
    G4double temperature = GetIceTemperatureAtPosition(position);
    
    // Step 2: Check molecule type and apply appropriate Arrhenius parameters
    // VANILLA: Uses fixed molConf->GetDiffusionCoefficient() for all molecules
    if (molConf->GetDefinition() == G4OH::Definition()) {
        // OH molecules: Higher activation energy (0.14 eV) = more temperature sensitive
        return ArrheniusConstants_ice::CalculateArrheniusDiffusion(
            ArrheniusConstants_ice::D0_OH, ArrheniusConstants_ice::Ea_OH, temperature);
    }
    else if (molConf->GetDefinition() == G4Hydrogen::Definition()) {
        // H molecules: Lower activation energy (0.02 eV) = less temperature sensitive
        return ArrheniusConstants_ice::CalculateArrheniusDiffusion(
            ArrheniusConstants_ice::D0_H, ArrheniusConstants_ice::Ea_H, temperature);
    }
    
    // Step 3: Fallback for other molecules (e-, e_aq, H3O+, etc.)
    // VANILLA: This is the only path vanilla takes for all molecules
    return molConf->GetDiffusionCoefficient();
}

// MAIN REACTION METHOD OVERRIDE
// This is where ice physics fundamentally differs from vanilla behavior
// VANILLA: Uses pMoleculeA->GetDiffusionCoefficient() and pMoleculeB->GetDiffusionCoefficient()
// ICE VERSION: Calculates Arrhenius coefficients in real-time based on track positions
std::unique_ptr<G4ITReactionChange> G4DNAMolecularReaction_ice::MakeReaction(const G4Track &trackA,
                                                                              const G4Track &trackB)
{
    // Step 1: Standard Geant4 reaction setup (identical to vanilla)
    std::unique_ptr<G4ITReactionChange> pChanges(new G4ITReactionChange());
    pChanges->Initialize(trackA, trackB);

    const auto pMoleculeA = GetMolecule(trackA)->GetMolecularConfiguration();
    const auto pMoleculeB = GetMolecule(trackB)->GetMolecularConfiguration();

    const auto pReactionData = fMolReactionTable->GetReactionData(pMoleculeA, pMoleculeB);

    const G4int nbProducts = pReactionData->GetNbProducts();

    if (nbProducts != 0)
    {
        // Step 2: ICE-SPECIFIC DIFFUSION COEFFICIENT CALCULATION
        // VANILLA DIFFERENCE: This is the critical change from vanilla implementation
        // VANILLA: const G4double D1 = pMoleculeA->GetDiffusionCoefficient();
        // VANILLA: const G4double D2 = pMoleculeB->GetDiffusionCoefficient();
        // ICE: Calculate position-dependent Arrhenius coefficients
        const G4double D1 = GetArrheniusDiffusionCoefficient(trackA, pMoleculeA);
        const G4double D2 = GetArrheniusDiffusionCoefficient(trackB, pMoleculeB);
        
        // Step 3: Reaction site calculation using ice-specific diffusion rates
        // PHYSICS: Reaction occurs closer to the slower-diffusing molecule
        // In cold ice regions, both molecules move slowly; in warm regions, both move faster
        // This creates realistic spatial distribution of chemical products
        const G4double sqrD1 = D1 == 0. ? 0. : std::sqrt(D1);
        const G4double sqrD2 = D2 == 0. ? 0. : std::sqrt(D2);
        const G4double inv_numerator = 1./(sqrD1 + sqrD2);
        const G4ThreeVector reactionSite = sqrD2 * inv_numerator * trackA.GetPosition()
                                         + sqrD1 * inv_numerator * trackB.GetPosition();

        // Step 4: Product creation (identical to vanilla after site calculation)
        for (G4int j = 0; j < nbProducts; ++j)
        {
            auto pProduct = new G4Molecule(pReactionData->GetProduct(j));
            auto pProductTrack = pProduct->BuildTrack(trackA.GetGlobalTime(), reactionSite);

            pProductTrack->SetTrackStatus(fAlive);

            G4ITTrackHolder::Instance()->Push(pProductTrack);

            pChanges->AddSecondary(pProductTrack);
            G4MoleculeFinder::Instance()->Push(pProductTrack);
        }
    }

    // Step 5: Cleanup (identical to vanilla)
    pChanges->KillParents(true);
    return pChanges;
}
